import os
import math
import numpy as np
from tqdm import tqdm
from enum import Enum
from loguru import logger
from copy import deepcopy
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Iterable

import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel

from .state import RunnerState
from .utils import listify, to_numpy, env_rank, AverageMeter, TimeMeter
from pytorch_tools.utils.visualization import plot_confusion_matrix
from pytorch_tools.utils.visualization import render_figure_to_tensor
from pytorch_tools.utils.tensorboard import CorrectedSummaryWriter as SummaryWriter


class Callback(object):
    """
    Abstract class that all callback(e.g., Logger) classes extends from.
    Must be extended before usage.
    usage example:
    begin
    ---epoch begin (one epoch - one run of every loader)
    ------loader begin
    ---------batch begin
    ---------on_after_backward
    ---------batch end
    ------loader end
    ---epoch end
    end
    """

    def __init__(self, *args, **kwargs):
        self.state = RunnerState()

    def set_state(self, state):
        self.state = state

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_loader_begin(self):
        pass

    def on_loader_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_begin(self):
        pass

    def on_end(self):
        pass

    def on_after_backward(self):
        """Called after ``loss.backward()`` but before optimizer does anything."""
        pass


class Callbacks(Callback):
    """Class that combines multiple callbacks into one. For internal use only"""

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = listify(callbacks)

    def set_state(self, state):
        for callback in self.callbacks:
            callback.set_state(state)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_loader_begin(self):
        for callback in self.callbacks:
            callback.on_loader_begin()

    def on_loader_end(self):
        for callback in self.callbacks:
            callback.on_loader_end()

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_after_backward(self):
        for callback in self.callbacks:
            callback.on_after_backward()


def rank_zero_only(cls: Callback) -> Callback:
    """Returns Identity callback if rank != zero. supports wrapping a class and an instance"""
    is_rank_zero = env_rank() == 0
    if isinstance(cls, type):  # cls is a class
        return cls if is_rank_zero else Callback
    else:  # cls is an instance of some Callback
        return cls if is_rank_zero else Callback()


class BatchMetrics(Callback):
    """
    Computes metrics values after each batch
    Args:
        metrics (List): Metrics to measure during training. All metrics
            must have `name` attribute.
    """

    def __init__(self, metrics):
        super().__init__()
        self.metrics = listify(metrics)
        self.metric_names = [m.name for m in self.metrics]

    def on_begin(self):
        for name in self.metric_names:
            self.state.metric_meters[name] = AverageMeter(name=name)

    @torch.no_grad()
    def on_batch_end(self):
        _, target = self.state.input
        output = self.state.output
        with amp.autocast(self.state.use_fp16):
            for metric, name in zip(self.metrics, self.metric_names):
                self.state.metric_meters[name].update(to_numpy(metric(output, target).squeeze()))


class LoaderMetrics(Callback):
    """
    Computes metrics values after running full loader
    Args:
        metrics (List): Metrics to measure during training. All metrics
            must have `name` attribute.
        update_steps (int): how ofter to recompute
        recompute_each_batch: Flag to compute metrics after each batch.
            Can be slow.
    """

    def __init__(self, metrics):
        super().__init__()
        self.metrics = listify(metrics)
        self.metric_names = [m.name for m in self.metrics]

        self.target = None
        self.output = None

    def on_begin(self):
        for name in self.metric_names:
            self.state.metric_meters[name] = AverageMeter(name=name)

    def on_loader_begin(self):
        self.target = []
        self.output = []

    def on_batch_end(self):
        _, target = self.state.input
        self.target.append(target.cpu().detach())
        self.output.append(self.state.output.cpu().detach())

    @torch.no_grad()
    def on_loader_end(self):

        target = torch.cat(self.target)
        output = torch.cat(self.output)
        with amp.autocast(self.state.use_fp16):
            for metric, name in zip(self.metrics, self.metric_names):
                self.state.metric_meters[name].update(to_numpy(metric(output, target).squeeze()))


@rank_zero_only
class Timer(Callback):
    """
    Profiles first epoch and prints time spend on data loader and on model.
    Usefull for profiling dataloader code performance
    """

    def __init__(self):
        super().__init__()
        self.has_printed = False
        self.timer = TimeMeter()

    def on_batch_begin(self):
        self.timer.batch_start()

    def on_batch_end(self):
        self.timer.batch_end()

    def on_loader_begin(self):
        self.timer.reset()

    def on_loader_end(self):
        if not self.has_printed:
            self.has_printed = True
            d_time = self.timer.data_time.avg_smooth
            b_time = self.timer.batch_time.avg_smooth
            logger.info(f"TimeMeter profiling. Data time: {d_time:.2E}s. Model time: {b_time:.2E}s \n")


@dataclass
class DataStage:
    # start and end epoch for current stage
    start: int = 0
    end: int = 1
    lr: "Optional[ (float, float) | float ]" = 1
    lr_mode: Optional[str] = "linear"
    # not processed by `PhasesScheduler` but allows more advanced logic
    extra_args: Optional[Dict] = None

    def __post_init__(self):
        # will pass for list, typle, omegaconf.listconfig
        if not isinstance(self.lr, Iterable):
            self.lr = [self.lr, self.lr]
        assert self.start < self.end, "start > end in stages"


class PhasesScheduler(Callback):
    """
    Scheduler that uses `phases` to process updates.
    Supported `mode`'s are {`linear`, `cos`, `poly`}
    NOTE: loss specified in phases would be multiplied by original loss in param groups

    Args:
        phases (List[DataStage]): phases
        change_every (int): how often to actually change the lr. changing too often may slowdown the training

    Example:
        PHASES = [
            {"start": 0, "end":8, "lr":[0,0.1]},
            {"start": 8, "end": 24, "lr":[0.1, 0.01], "lr_mode":"cos"},
            {"start": 24, "end": 30, "lr": 0.001},
        ]
    """

    def __init__(self, phases: List[DataStage], change_every=50):
        if type(phases[0]) != DataStage:
            phases = [DataStage(**ph) for ph in phases]

        self.phases = phases
        self.change_every = change_every
        self.current_lr = None
        self.tot_epochs = max([p.end for p in phases])
        self.phase = self.phases[0]
        super(PhasesScheduler, self).__init__()

    @staticmethod
    def _schedule(start: int, end: int, pct: float, mode: str):
        """anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        if mode == "linear":
            return start + (end - start) * pct
        elif mode == "cos":
            return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)
        elif mode == "poly":
            gamma = (end / start) ** (1 / 100)
            return start * gamma ** (pct * 100)
        else:
            raise ValueError(f"Mode: `{mode}` is not supported in PhasesScheduler")

    def _get_lr(self, batch_curr: int) -> float:
        phase = self.phase
        batch_tot = self.state.epoch_size
        ep_curr, ep_tot = self.state.epoch - phase.start, phase.end - phase.start
        perc = (ep_curr * batch_tot + batch_curr) / ((ep_tot + 1) * batch_tot)
        assert perc <= 1 and perc >= 0
        lr_start, lr_end = phase.lr  # may be the same
        new_lr = self._schedule(lr_start, lr_end, perc, phase.lr_mode)
        return new_lr

    def on_begin(self):
        self.original_lr = [pg.get("lr", 1) for pg in self.state.optimizer.param_groups]

    def on_epoch_begin(self):
        new_phase = None
        for phase in reversed(self.phases):
            if self.state.epoch >= phase.start:
                new_phase = phase
                break
        if new_phase is None or self.state.epoch > phase.end:
            raise ValueError("Epoch out of range")
        else:
            self.phase = new_phase

    def on_batch_begin(self):
        lr = self._get_lr(self.state.step)
        if self.current_lr == lr or (self.state.step % self.change_every != 0):
            return
        self.current_lr = lr
        for orig_lr, param_group in zip(self.original_lr, self.state.optimizer.param_groups):
            param_group["lr"] = lr * orig_lr


class ReduceMode(Enum):
    MIN = "min"
    MAX = "max"


class ReduceLROnPlateau(Callback):
    """
    Reduces learning rate by `factor` after `patience`epochs without improving

    Args:
        factor (float): by how to reduce learning rate
        patience (int): how many epochs to wait until reducing lr
        min_lr (float): minimum learning rate which could be achieved
        monitor (str): quantity to monitor. Implicitly prefers validation metrics over train. One of:
            `loss` or name of any metric passed to the runner.
        mode (str): one of "min" of "max". Whether to decide reducing based
            on minimizing or maximizing loss
        vebose (bool): Whether or not to print messages about updating lr to console
    """

    def __init__(self, factor=0.5, patience=5, min_lr=1e-6, monitor="loss", mode="min", verbose=True):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.monitor = monitor
        self.min_lr = min_lr
        mode = ReduceMode(mode)
        self.best = np.inf if mode == ReduceMode.MIN else -np.inf
        self.monitor_op = np.less if mode == ReduceMode.MIN else np.greater
        self._steps_since_best = 0
        self.verbose = verbose

    def on_epoch_end(self):
        current = self.get_monitor_value()
        self._steps_since_best += 1
        if self.monitor_op(current, self.best):
            self._steps_since_best = 0
        elif self._steps_since_best > self.patience:
            for param_group in self.state.optimizer.param_groups:
                if param_group["lr"] * self.factor > self.min_lr:
                    param_group["lr"] *= self.factor
            if self.verbose:
                logger.info(f"ReduceLROnPlateau reducing learning rate to {param_group['lr'] * self.factor}")

    def get_monitor_value(self):
        value = None
        if self.monitor == "loss":
            value = self.state.loss_meter.avg
        else:
            for name, metric_meter in self.state.metric_meters.items():
                if name == self.monitor:
                    value = metric_meter.avg
        if value is None:
            raise ValueError(f"ReduceLROnPlateau can't find {self.monitor} value to monitor")
        return value


@rank_zero_only
class CheckpointSaver(Callback):
    """
    Save best model every epoch based on loss

    Args:
        save_dir (str): path to folder where to save the model
        save_name (str): name of the saved model. can additionally
            add epoch and metric to model save name
        monitor (str): quantity to monitor. Implicitly prefers validation metrics over train. One of:
            `loss` or name of any metric passed to the runner.
        mode (str): one of "min" of "max". Whether to decide to save based
            on minimizing or maximizing loss
        include_optimizer (bool): if True would also save `optimizers` state_dict.
            This increases checkpoint size 2x times.
        verbose (bool): If `True` reports each time new best is found
    """

    def __init__(
        self,
        save_dir,
        save_name="model_{ep}_{metric:.2f}.chpn",
        monitor="loss",
        mode="min",
        include_optimizer=False,
        verbose=True,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.monitor = monitor
        mode = ReduceMode(mode)
        if mode == ReduceMode.MIN:
            self.best = np.inf
            self.monitor_op = np.less
        elif mode == ReduceMode.MAX:
            self.best = -np.inf
            self.monitor_op = np.greater
        self.include_optimizer = include_optimizer
        self.verbose = verbose

    def on_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current, self.best):
            ep = self.state.epoch_log
            if self.verbose:
                logger.info(f"Epoch {ep:2d}: best {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current
            save_name = os.path.join(self.save_dir, self.save_name.format(ep=ep, metric=current))
            self._save_checkpoint(save_name)

    def _save_checkpoint(self, path):
        if hasattr(self.state.model, "module"):  # used for saving DDP models
            state_dict = self.state.model.module.state_dict()
        else:
            state_dict = self.state.model.state_dict()
        save_dict = {"epoch": self.state.epoch, "state_dict": state_dict}
        if self.include_optimizer:
            save_dict["optimizer"] = self.state.optimizer.state_dict()
        torch.save(save_dict, path)

    def get_monitor_value(self):
        value = None
        if self.monitor == "loss":
            value = self.state.loss_meter.avg
        else:
            for name, metric_meter in self.state.metric_meters.items():
                if name == self.monitor:
                    value = metric_meter.avg
        if value is None:
            raise ValueError(f"CheckpointSaver can't find {self.monitor} value to monitor")
        return value


@rank_zero_only
class TensorBoard(Callback):
    """
    Saves training and validation statistics for TensorBoard

    Args:
        log_dir (str): path where to store logs
        log_every (int): how often (in batches) to write logs during training
    """

    def __init__(self, log_dir=None, log_every=20):
        super().__init__()
        self.log_dir = log_dir
        self.log_every = log_every

    def on_begin(self):
        if self.state.tb_logger is not None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        self.state.tb_logger = SummaryWriter(self.log_dir)

    def on_batch_end(self):
        if self.state.is_train and (self.state.step % self.log_every == 0):
            # TODO: select better name instead of train_ ?
            self.state.tb_logger.add_scalar(
                "train_/loss", self.state.loss_meter.avg_smooth, self.state.global_sample_step
            )
            for name, metric in self.state.metric_meters.items():
                self.state.tb_logger.add_scalar(f"train_/{name}", metric.avg_smooth, self.state.global_sample_step)

    def on_epoch_end(self):
        self.state.tb_logger.add_scalar("train/loss", self.state.train_loss.avg, self.state.global_sample_step)
        for name, metric in self.state.train_metrics.items():
            self.state.tb_logger.add_scalar(f"train/{name}", metric.avg, self.state.global_sample_step)

        lr = sorted([pg["lr"] for pg in self.state.optimizer.param_groups])[-1]  # largest lr
        self.state.tb_logger.add_scalar("train_/lr", lr, self.state.global_sample_step)
        self.state.tb_logger.add_scalar("train/epoch", self.state.epoch, self.state.global_sample_step)
        # don't log if no val
        if self.state.val_loss is None:
            return

        self.state.tb_logger.add_scalar("val/loss", self.state.val_loss.avg, self.state.global_sample_step)
        for name, metric in self.state.val_metrics.items():
            self.state.tb_logger.add_scalar(f"val/{name}", metric.avg, self.state.global_sample_step)

    def on_end(self):
        self.state.tb_logger.close()


class TensorBoardCM(Callback):
    """
    Saves training and validation Confusion Matrix as image

    Args:
        class_namess (List[str]): list of class names for proper visualization
    """

    def __init__(self, class_names=None):
        self.class_names = class_names
        self.n_classes = None  # will infer implicitly later
        self.cmap = None
        self.train_cm_img = None
        self.val_cm_img = None

    def on_batch_end(self):
        if self.cmap is None:
            self.n_classes = self.state.output.shape[1]
            self.cmap = np.zeros((self.n_classes, self.n_classes), dtype=int)
            if self.class_names is None:
                self.class_names = [str(i) for i in range(self.n_classes)]

        target = self.state.input[1]
        if len(target.shape) == 2:
            target = target.argmax(1)
        predict = self.state.output.argmax(1)
        for tr, pr in zip(target, predict):
            self.cmap[tr, pr] += 1

    def on_loader_end(self):
        f = plot_confusion_matrix(self.cmap, self.class_names, normalize=True, show=False)
        cm_img = render_figure_to_tensor(f)
        if self.state.is_train:
            self.train_cm_img = cm_img
        else:
            self.val_cm_img = cm_img
        self.cmap = None

    def on_epoch_end(self):
        if self.train_cm_img is not None:
            self.state.tb_logger.add_image("train/confusion_matrix", self.train_cm_img, self.state.global_sample_step)
        if self.val_cm_img is not None:
            self.state.tb_logger.add_image("val/confusion_matrix", self.val_cm_img, self.state.global_sample_step)


@rank_zero_only
class ConsoleLogger(Callback):
    """Prints training progress to console for monitoring."""

    def on_loader_begin(self):
        if hasattr(tqdm, "_instances"):  # prevents many printing issues
            tqdm._instances.clear()
        stage_str = "train" if self.state.is_train else "validat"
        desc = f"Epoch {self.state.epoch_log:2d}/{self.state.num_epochs}. {stage_str}ing"
        self.pbar = tqdm(total=self.state.epoch_size, desc=desc, ncols=0)

    def on_loader_end(self):
        # update to avg
        desc = OrderedDict({"Loss": f"{self.state.loss_meter.avg:.4f}"})
        desc.update({name: f"{m.avg:.3f}" for (name, m) in self.state.metric_meters.items()})
        self.pbar.set_postfix(**desc)
        self.pbar.update()
        self.pbar.close()

    def on_batch_end(self):
        desc = OrderedDict({"Loss": f"{self.state.loss_meter.avg_smooth:.4f}"})
        desc.update({name: f"{m.avg:.3f}" for (name, m) in self.state.metric_meters.items()})
        self.pbar.set_postfix(**desc)
        self.pbar.update()


class FileLogger(Callback):
    """Logs loss and metrics every epoch.
    You have to manually configure loguru.logger to enable actual logging to file"""

    def on_epoch_begin(self):
        logger.info(f"Epoch {self.state.epoch_log} | lr {self.current_lr:.2e}")

    def on_epoch_end(self):
        loss, metrics = self.state.train_loss, self.state.train_metrics
        logger.info("Train " + self._format_meters(loss, metrics))
        if self.state.val_loss is not None:
            loss, metrics = self.state.val_loss, self.state.val_metrics
            logger.info("Val   " + self._format_meters(loss, metrics))

    @property
    def current_lr(self):
        return sorted([pg["lr"] for pg in self.state.optimizer.param_groups])[-1]

    @staticmethod
    def _format_meters(loss, metrics):
        return f"loss: {loss.avg:.4f} | " + " | ".join(f"{m.name}: {m.avg:.4f}" for m in metrics.values())


class GradientClipping(Callback):
    """Clips gradients by max norm"""

    def __init__(self, gradient_clip_val):
        self.gradient_clip_val = gradient_clip_val
        assert gradient_clip_val > 0, "Invalid value for gradient clipping"

    def on_after_backward(self):
        if self.state.step % self.state.accumulate_steps != 0:
            return
        self.state.grad_scaler.unscale_(self.state.optimizer)
        torch.nn.utils.clip_grad_norm_(self.state.model.parameters(), self.gradient_clip_val)


class AdaptiveGradientClipping(Callback):
    """Clips gradient based on max norm.

    Ref:
        High-Performance Large-Scale Image Recognition Without Normalization
        Official JAX impl (paper authors): https://github.com/deepmind/deepmind-research/tree/master/nfnets
        Phil Wang's PyTorch gist: https://gist.github.com/lucidrains/0d6560077edac419ab5d3aa29e674d5c
        Timm code: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
    """

    def __init__(self, clip_factor, eps=1e-3):
        self.clip_factor = clip_factor
        self.eps = eps

    @staticmethod
    def _unitwise_norm(x, norm_type=2.0):
        if x.ndim <= 1:
            return x.norm(norm_type)
        else:
            # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
            # might need special cases for other weights (possibly MHA) where this may not be true
            return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)

    def on_after_backward(self):
        if self.state.step % self.state.accumulate_steps != 0:
            return

        self.state.grad_scaler.unscale_(self.state.optimizer)

        for p in self.state.model.parameters():
            if p.grad is None:
                continue
            p_data = p.detach()
            g_data = p.grad.detach()
            max_norm = self._unitwise_norm(p_data).clamp_(min=self.eps).mul_(self.clip_factor)
            grad_norm = self._unitwise_norm(g_data)
            clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
            new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
            p.grad.detach().copy_(new_grads)


class Mixup(Callback):
    """Performs mixup on input. Only for classification.
    Mixup blends two images by linear interpolation between them with small
    lambda drown from Beta distribution. Labels become a liner blend of true labels for
    original images.
    Ref: https://arxiv.org/abs/1710.09412
    According to https://arxiv.org/abs/2001.06268 mixing images from different batches give better results

    Args:
        alpha (float): hyperparameter from paper. Suggested default is 0.2 for Imagenet.
        num_classes (int): number of classes. Mixup implicitly turns
            labels into one hot encoding and needs number of classes to do that
        prob (float): probability of applying mixup
    """

    def __init__(self, alpha, num_classes, prob=0.5):
        super().__init__()
        self.tb = torch.distributions.Beta(alpha, alpha)
        self.num_classes = num_classes
        self.prob = prob
        self.prev_input = None

    def on_batch_begin(self):
        self.state.input = self.mixup(*self.state.input)

    @torch.no_grad()
    def mixup(self, data, target):
        if len(target.shape) == 1:  # if not one hot
            target_one_hot = torch.zeros(target.size(0), self.num_classes, dtype=torch.float, device=data.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        else:
            target_one_hot = target
        if not self.state.is_train or np.random.rand() > self.prob:
            return data, target_one_hot
        prev_data, prev_target = (data, target_one_hot) if self.prev_input is None else self.prev_input
        self.prev_input = data.clone(), target_one_hot.clone()
        perm = torch.randperm(data.size(0), device=data.device)
        c = self.tb.sample()
        md = c * data + (1 - c) * prev_data[perm]
        mt = c * target_one_hot + (1 - c) * prev_target[perm]
        return md, mt


class Cutmix(Callback):
    """Performs CutMix on input. Only for classification.
    Cutmix combines two images by replacing part of the image by part from another image.
    New label is proportional to area of inserted part.
    Ref: https://arxiv.org/abs/1905.04899
    According to https://arxiv.org/abs/2001.06268 mixing images from different batches give better results

    Args:
        alpha (float): hyperparameter from paper. Suggested default is 1.0 for Imagenet.
        num_classes (int): number of classes. CutMix implicitly turns
            labels into one hot encoding and needs number of classes to do that
        prob (float): probability of applying cutmix
    """

    def __init__(self, alpha, num_classes, prob=0.5):
        super().__init__()
        self.tb = torch.distributions.Beta(alpha, alpha)
        self.num_classes = num_classes
        self.prob = prob
        self.prev_input = None

    def on_batch_begin(self):
        self.state.input = self.cutmix(*self.state.input)

    def on_loader_begin(self):
        self.prev_input = None  # avoid leak from test to train

    @torch.no_grad()
    def cutmix(self, data, target):
        if target.dim() == 1:  # if not one hot
            target_one_hot = torch.zeros(target.size(0), self.num_classes, dtype=torch.float, device=data.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        else:
            target_one_hot = target
        if not self.state.is_train or np.random.rand() > self.prob:
            return data, target_one_hot
        prev_data, prev_target = (data, target_one_hot) if self.prev_input is None else self.prev_input
        self.prev_input = data.clone(), target_one_hot.clone()
        # prev_data shape can be different from current. so need to take min
        H, W = min(data.size(2), prev_data.size(2)), min(data.size(3), prev_data.size(3))
        perm = torch.randperm(data.size(0), device=data.device)
        lam = self.tb.sample()
        lam = min([lam, 1 - lam])
        bbh1, bbw1, bbh2, bbw2 = self.rand_bbox(H, W, lam)
        # real lambda may be diffrent from sampled. adjust for it
        lam = (bbh2 - bbh1) * (bbw2 - bbw1) / (H * W)
        data[:, :, bbh1:bbh2, bbw1:bbw2] = prev_data[perm, :, bbh1:bbh2, bbw1:bbw2]
        mixed_target = (1 - lam) * target_one_hot + lam * prev_target[perm]
        return data, mixed_target

    @staticmethod
    def rand_bbox(H, W, lam):
        """returns bbox with area close to lam*H*W"""
        cut_rat = np.sqrt(lam)
        cut_h = np.int(H * cut_rat)
        cut_w = np.int(W * cut_rat)
        # uniform
        ch = np.random.randint(H)
        cw = np.random.randint(W)
        bbh1 = np.clip(ch - cut_h // 2, 0, H)
        bbw1 = np.clip(cw - cut_w // 2, 0, W)
        bbh2 = np.clip(ch + cut_h // 2, 0, H)
        bbw2 = np.clip(cw + cut_w // 2, 0, W)
        return bbh1, bbw1, bbh2, bbw2


class SegmCutmix(Cutmix):
    """Cutmix for segmentation tasks. see `Cutmix` for more details"""

    def __init__(self, alpha=1.0, prob=0.5):
        super().__init__(alpha, None, prob)

    @torch.no_grad()
    def cutmix(self, data, target):
        if not self.state.is_train or np.random.rand() > self.prob:
            return data, target
        prev_data, prev_target = (data, target) if self.prev_input is None else self.prev_input
        self.prev_input = data.clone(), target.clone()
        H, W = min(data.size(2), prev_data.size(2)), min(data.size(3), prev_data.size(3))
        perm = torch.randperm(data.size(0), device=data.device)
        lam = self.tb.sample()
        lam = min([lam, 1 - lam])
        bbh1, bbw1, bbh2, bbw2 = self.rand_bbox(H, W, lam)
        data[:, :, bbh1:bbh2, bbw1:bbw2] = prev_data[perm, :, bbh1:bbh2, bbw1:bbw2]
        target[:, :, bbh1:bbh2, bbw1:bbw2] = prev_target[perm, :, bbh1:bbh2, bbw1:bbw2]
        return data, target


class ScheduledDropout(Callback):
    def __init__(self, drop_rate=0.1, epochs=30, attr_name="dropout.p"):
        """
        Slowly changes dropout value for `attr_name` each epoch.
        Ref: https://arxiv.org/abs/1703.06229
        Args:
            drop_rate (float): max dropout rate
            epochs (int): num epochs to max dropout to fully take effect
            attr_name (str): name of dropout block in model
        """
        super().__init__()
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.attr_name = attr_name

    def on_epoch_end(self):
        current_rate = self.drop_rate * min(1, self.state.epoch / self.epochs)
        setattr(self.state.model, self.attr_name, current_rate)


class ResetOptimizer(Callback):
    """Set's Optimizers state to empty for epoch in `reset_epoch`. Could be used for restarts.
    Args:
        reset_epoch (List[int]): after which epochs to reset optimizer
        verbose (bool): Flag to print that optimizer was reset."""

    def __init__(self, reset_epochs=[], verbose=True):
        super().__init__()
        self.reset_epochs = set(reset_epochs)
        self.verbose = verbose

    def on_epoch_end(self):
        if self.state.epoch_log in self.reset_epochs:
            # any optimizer inherited from torch.Optimizer has state which can be reset
            if hasattr(self.state.optimizer, "optimizer"):  # for lookahead
                self.state.optimizer.optimizer.state = defaultdict(dict)
            else:
                self.state.optimizer.state = defaultdict(dict)

            if self.verbose:
                logger.info("Reseting optimizer")


# docstring from https://github.com/rwightman/pytorch-image-models
class ModelEma(Callback):
    """Model Exponential Moving Average
    Keeps a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results.

    Current implementation follows TensorFlow and uses the following formula:
    ema -= (1 - decay) * (ema - model)
    This is mathematically equivalent to the classic formula below but inplace is faster
    ema = decay * ema + (1 - decay) * model

    NOTE: Pay attention to the decay constant you are using relative to your update count per epoch.

    NOTE: put this Callback AFTER Checkpoint saver! Otherwise you would validate EMA weights but save
    model weights

    NOTE: Need to be used in all process (not only master)! otherwise you would save not the best model bur some random

    NOTE: Pass model to ModelEma after cuda() and AMP but before SyncBN and DDP wrapper

    Args:
        model (nn.Module): model after cuda and AMP
        decay (float): decay for EMA for every step
        decay_every (int): how oftern to really decay weights. Decaying every step produced a
            visible training slowdown. Real decay factor is adjusted to match every step update.
    """

    def __init__(self, decay=0.9999, decay_every=10):
        super().__init__()
        assert isinstance(decay, float), "Decay in EMA should be float"
        self.ema = None
        self.model_copy = None
        self.decay_factor = 1 - decay**decay_every  # simulate every step decay
        self.decay_every = decay_every

    def on_begin(self):
        if self.ema is not None:
            return
        model = self.state.model
        if isinstance(self.state.model, DistributedDataParallel):
            model = model.module
        self.ema = deepcopy(model).eval().requires_grad_(False)
        self.state.communication_dict["ema_model"] = self.ema

    def on_batch_end(self):
        if not self.state.is_train or (self.state.step % self.decay_every != 0):
            return

        with torch.no_grad():
            for (ema_v, m_v) in zip(self.ema.state_dict().values(), self.state.model.state_dict().values()):
                if m_v.dtype == torch.long:  # to prevent errors on `num_batches_tracked` in BN
                    continue
                ema_v.sub_(ema_v.sub(m_v), alpha=self.decay_factor)

    def on_loader_begin(self):
        if self.state.is_train:
            return
        # validate on ema model
        self.model_copy = self.state.model
        self.state.model = self.ema

    def on_epoch_end(self):
        if self.state.is_train:
            return
        # return model back
        self.state.model = self.model_copy


class BatchOverfit(Callback):
    """Remembers first batch and tries to overfit it. Useful for debug.
    NOTE: Should go after all other callbacks to make sure it's the last thing to change the input
    Args:
        save_batch (bool): If True will save first batch. Useful for visualization"""

    def __init__(self, save_batch=False):
        super().__init__()
        self.has_saved = False
        self.save_batch = save_batch
        self.batch = None

    def on_batch_begin(self):
        if not self.has_saved:
            self.has_saved = True
            self.batch = self.state.input[0].clone(), self.state.input[1].clone()
            if self.save_batch:
                torch.save(self.batch[0], "b_img")
                torch.save(self.batch[1], "b_target")
        else:
            self.state.input = self.batch
