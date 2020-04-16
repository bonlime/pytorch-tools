import os
import math
import logging
from tqdm import tqdm
from enum import Enum
from collections import OrderedDict
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .state import RunnerState
from pytorch_tools.utils.misc import listify
from pytorch_tools.utils.visualization import plot_confusion_matrix
from pytorch_tools.utils.visualization import render_figure_to_tensor


class Callback(object):
    """
    Abstract class that all callback(e.g., Logger) classes extends from.
    Must be extended before usage.
    usage example:
    begin
    ---epoch begin (one epoch - one run of every loader)
    ------loader begin 
    ---------batch begin
    ---------batch end
    ------loader end 
    ---epoch end
    end
    """

    def __init__(self):
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


class Timer(Callback):
    """
    Profiles first epoch and prints time spend on data loader and on model.
    Usefull for profiling dataloader code performance
    """

    def __init__(self):
        super().__init__()
        self.has_printed = False

    def on_batch_begin(self):
        self.state.timer.batch_start()

    def on_batch_end(self):
        self.state.timer.batch_end()

    def on_loader_end(self):
        if not self.has_printed:
            self.has_printed = True
            d_time = self.state.timer.data_time.avg_smooth
            b_time = self.state.timer.batch_time.avg_smooth
            print(f"\nTimeMeter profiling. Data time: {d_time:.2E}s. Model time: {b_time:.2E}s \n")


class PhasesScheduler(Callback):
    """
    Scheduler that uses `phases` to process updates.

    Args:
        phases (List[Dict]): phases
        change_every (int): how often to actually change the lr. changing too 
            often may slowdown the training

    Example:
        PHASES = [
            {"ep":[0,8],  "lr":[0,0.1], "mom":0.9, },
            {"ep":[8,24], "lr":[0.1, 0.01], "mode":"cos"},
            {'ep':[24, 30], "lr": 0.001},
        ]
    """

    def __init__(self, phases, change_every=50):
        self.change_every = change_every
        self.current_lr = None
        self.current_mom = None
        self.phases = [self._format_phase(p) for p in phases]
        self.phase = self.phases[0]
        self.tot_epochs = max([max(p["ep"]) for p in self.phases])
        super(PhasesScheduler, self).__init__()

    def _format_phase(self, phase):
        phase["ep"] = listify(phase["ep"])
        phase["lr"] = listify(phase["lr"])
        phase["mom"] = listify(phase.get("mom", None))  # optional
        if len(phase["lr"]) == 2 or len(phase["mom"]) == 2:
            phase["mode"] = phase.get("mode", "linear")
            assert len(phase["ep"]) == 2, "Linear learning rates must contain end epoch"
        return phase

    @staticmethod
    def _schedule(start, end, pct, mode):
        """anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        if mode == "linear":
            return start + (end - start) * pct
        elif mode == "cos":
            return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)

    def _get_lr_mom(self, batch_curr):
        phase = self.phase
        batch_tot = self.state.epoch_size
        if len(phase["ep"]) == 1:
            perc = 0
        else:
            ep_start, ep_end = phase["ep"]
            ep_curr, ep_tot = self.state.epoch - ep_start, ep_end - ep_start
            perc = (ep_curr * batch_tot + batch_curr) / (ep_tot * batch_tot)
        if len(phase["lr"]) == 1:
            new_lr = phase["lr"][0]  # constant learning rate
        else:
            lr_start, lr_end = phase["lr"]
            new_lr = self._schedule(lr_start, lr_end, perc, phase["mode"])

        if len(phase["mom"]) == 0:
            new_mom = self.current_mom
        elif len(phase["mom"]) == 1:
            new_mom = phase["mom"][0]
        else:
            mom_start, mom_end = phase["mom"]
            new_mom = self._schedule(mom_start, mom_end, perc, phase["mode"])

        return new_lr, new_mom

    def on_epoch_begin(self):
        new_phase = None
        for phase in reversed(self.phases):
            if self.state.epoch >= phase["ep"][0]:
                new_phase = phase
                break
        if new_phase is None:
            raise Exception("Epoch out of range")
        else:
            self.phase = new_phase

    def on_batch_begin(self):
        lr, mom = self._get_lr_mom(self.state.step)
        if (self.current_lr == lr and self.current_mom == mom) or (self.state.step % self.change_every != 0):
            return
        self.current_lr = lr
        self.current_mom = mom
        for param_group in self.state.optimizer.param_groups:
            param_group["lr"] = lr
            param_group["momentum"] = mom


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
        mode (str): one of "min" of "max". Whether to decide reducing based
            on minimizing or maximizing loss
        vebose (bool): Whether or not to print messages about updating lr to console
    """

    def __init__(self, factor=0.5, patience=5, min_lr=1e-6, mode="min", verbose=True):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.mode = ReduceMode(mode)
        self.min_lr = min_lr
        self.best = float("inf") if self.mode == ReduceMode.MIN else -float("inf")
        self._steps_since_best = 0
        self.verbose = verbose

    def on_epoch_end(self):
        # TODO: zakirov(19.11.19) Add support for saving based on metric
        if self.state.val_loss is not None:
            current = self.state.val_loss.avg
        else:
            current = self.state.train_loss.avg
        self._steps_since_best += 1

        if (self.mode == ReduceMode.MIN and current < self.best) or (
            self.mode == ReduceMode.MAX and current > self.best
        ):
            self._steps_since_best = 0
        elif self._steps_since_best > self.patience:
            for param_group in self.state.optimizer.param_groups:
                if param_group["lr"] * self.factor > self.min_lr:
                    param_group["lr"] *= self.factor
            if self.verbose:
                print(f"ReduceLROnPlateau reducing learning rate to {param_group['lr'] * self.factor}")


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
                print(f"Epoch {ep:2d}: best {self.monitor} improved from {self.best:.4f} to {current:.4f}")
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
            for metric_meter in self.state.metric_meters:
                if metric_meter.name == self.monitor:
                    value = metric_meter.avg
        if value is None:
            raise ValueError(f"CheckpointSaver can't find {self.monitor} value to monitor")
        return value


class TensorBoard(Callback):
    """
    Saves training and validation statistics for TensorBoard

    Args:
        log_dir (str): path where to store logs
        log_every (int): how often to write logs during training
    """

    def __init__(self, log_dir, log_every=20):
        super().__init__()
        self.log_dir = log_dir
        self.log_every = log_every
        self.writer = None
        self.current_step = 0

    def on_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_batch_end(self):
        self.current_step += self.state.batch_size
        if self.state.is_train and (self.current_step % self.log_every == 0):
            self.writer.add_scalar(
                # need proper name
                "train_/loss",
                self.state.loss_meter.val,
                self.current_step,
            )
            for m in self.state.metric_meters:
                self.writer.add_scalar(f"train_/{m.name}", m.val, self.current_step)

    def on_epoch_end(self):
        self.writer.add_scalar("train/loss", self.state.train_loss.avg, self.current_step)
        for m in self.state.train_metrics:
            self.writer.add_scalar(f"train/{m.name}", m.avg, self.current_step)

        lr = sorted([pg["lr"] for pg in self.state.optimizer.param_groups])[-1]  # largest lr
        # TODO: select better name instead of train_ ?
        self.writer.add_scalar("train_/lr", lr, self.current_step)
        self.writer.add_scalar("train/epoch", self.state.epoch, self.current_step)
        # don't log if no val
        if self.state.val_loss is None:
            return

        self.writer.add_scalar("val/loss", self.state.val_loss.avg, self.current_step)
        for m in self.state.val_metrics:
            self.writer.add_scalar(f"val/{m.name}", m.avg, self.current_step)

    def on_end(self):
        self.writer.close()


class TensorBoardWithCM(TensorBoard):
    """
    Saves training and validation statistics for TensorBoard
    And also saves Confusion Matrix as image

    Args:
        log_dir (str): path where to store logs
        log_every (int): how often to write logs during training
        class_namess (List[str]): list of class names for proper visualization
    """

    def __init__(self, log_dir, log_every=20, class_names=None):
        super().__init__(log_dir, log_every)
        self.class_names = class_names
        self.n_classes = None  # will infer implicitly later
        self.cmap = None
        self.train_cm_img = None
        self.val_cm_img = None

    def on_batch_end(self):
        super().on_batch_end()
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
        super().on_loader_end()
        f = plot_confusion_matrix(self.cmap, self.class_names, normalize=True, show=False)
        cm_img = render_figure_to_tensor(f)
        if self.state.is_train:
            self.train_cm_img = cm_img
        else:
            self.val_cm_img = cm_img
        self.cmap = None

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.train_cm_img is not None:
            self.writer.add_image("train/confusion_matrix", self.train_cm_img, self.current_step)
        if self.val_cm_img is not None:
            self.writer.add_image("val/confusion_matrix", self.val_cm_img, self.current_step)


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
        desc.update({m.name: f"{m.avg:.3f}" for m in self.state.metric_meters})
        self.pbar.set_postfix(**desc)
        self.pbar.update()
        self.pbar.close()

    def on_batch_end(self):
        desc = OrderedDict({"Loss": f"{self.state.loss_meter.avg_smooth:.4f}"})
        desc.update({m.name: f"{m.avg_smooth:.3f}" for m in self.state.metric_meters})
        self.pbar.set_postfix(**desc)
        self.pbar.update()


class FileLogger(Callback):
    """Logs loss and metrics every epoch into file
    Args:
        log_dir (str): path where to store the logs
        logger (logging.Logger): external logger. Default None
    """

    def __init__(self, log_dir, logger=None):
        # logger - already created instance of logger
        super().__init__()
        self.logger = logger or self._get_logger(os.path.join(log_dir, "logs.txt"))

    def on_epoch_begin(self):
        self.logger.info(f"Epoch {self.state.epoch_log} | lr {self.current_lr:.3f}")

    def on_epoch_end(self):
        loss, metrics = self.state.train_loss, self.state.train_metrics
        self.logger.info("Train " + self._format_meters(loss, metrics))
        if self.state.val_loss is not None:
            loss, metrics = self.state.val_loss, self.state.val_metrics
            self.logger.info("Val   " + self._format_meters(loss, metrics))

    @staticmethod
    def _get_logger(log_path):
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @property
    def current_lr(self):
        return sorted([pg["lr"] for pg in self.state.optimizer.param_groups])[-1]

    @staticmethod
    def _format_meters(loss, metrics):
        return f"loss: {loss.avg:.4f} | " + " | ".join(f"{m.name}: {m.avg:.4f}" for m in metrics)


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
            target_one_hot = torch.zeros(
                target.size(0), self.num_classes, dtype=torch.float, device=data.device
            )
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
        prob (float): probability of applying mixup
    """

    def __init__(self, alpha, num_classes, prob=0.5):
        super().__init__()
        self.tb = torch.distributions.Beta(alpha, alpha)
        self.num_classes = num_classes
        self.prob = prob
        self.prev_input = None

    def on_batch_begin(self):
        self.state.input = self.cutmix(*self.state.input)

    @torch.no_grad()
    def cutmix(self, data, target):
        if len(target.shape) == 1:  # if not one hot
            target_one_hot = torch.zeros(
                target.size(0), self.num_classes, dtype=torch.float, device=data.device
            )
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
        """ returns bbox with area close to lam*H*W """
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
            if hasattr(self.state.optimizer, "optimizer"): # for lookahead
                self.state.optimizer.optimizer.state = defaultdict(dict)
            else:
                self.state.optimizer.state = defaultdict(dict)
                
            if self.verbose:
                print("Reseting optimizer")