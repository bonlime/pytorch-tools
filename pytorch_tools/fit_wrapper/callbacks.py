import os
import math
import logging
from enum import Enum
import torch
from torch.utils.tensorboard import SummaryWriter
from ..utils.misc import listify
from .state import RunnerState


class Callback(object):
    """
    Abstract class that all callback(e.g., Logger) classes extends from.
    Must be extended before usage.
    usage example:
    train start
    ---epoch start (one epoch - one run of every loader)
    ------batch start
    ------batch handler
    ------batch end
    ---epoch end
    train end
    """

    def __init__(self):
        self.runner = None
        self.metrics = None
        self.state = RunnerState()

    def set_state(self, state):
        self.state = state

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks(Callback):
    """Class that combines multiple callbacks into one. For internal use only"""
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = listify(callbacks)

    def set_state(self, state):
        super().set_state(state)
        for callback in self.callbacks:
            callback.set_state(state)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


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

    def on_epoch_end(self):
        if not self.has_printed and self.state.verbose:
            self.has_printed = True
            d_time = self.state.timer.data_time.avg_smooth
            b_time = self.state.timer.batch_time.avg_smooth
            print(f"TimeMeter profiling. Data time: {d_time}. Model time: {b_time}")


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
        batch_tot = self.state.ep_size
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
        if (self.current_lr == lr and self.current_mom == mom) or (
            self.state.step % self.change_every != 0
        ):
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
        
    """

    def __init__(self, factor=0.5, patience=5, min_lr=1e-6, mode="min"):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf") if mode == ReduceMode.MIN else -float("inf")
        self._steps_since_best = 0
        self.mode = ReduceMode(mode)

    def on_epoch_end(self):
        # TODO: zakirov(19.11.19) Add support for saving based on metric
        # TODO: zakirov(20.12.19) Add some logging when lr is reduced
        current = self.state.val_loss.avg or self.state.train_loss.avg
        self._steps_since_best += 1
        
        if (self.mode == ReduceMode.MIN and current < self.best) or (
            self.mode == ReduceMode.MAX and current > self.best
        ):
            self._steps_since_best = 0
        elif self._steps_since_best > self.patience:
            for param_group in self.state.optimizer.param_groups:
                if param_group["lr"] * self.factor > self.min_lr:
                    param_group["lr"] *= self.factor


class CheckpointSaver(Callback):
    """
    Save best model every epoch based on loss
    
    Args:
        save_dir (str): path to folder where to save the model
        save_name (str): name of the saved model. can additionally 
            add epoch and metric to model save name
        mode (str): one of "min" of "max". Whether to decide to save based
            on minimizing or maximizing loss
    """
    def __init__(self, save_dir, save_name="model_{ep}_{metric:.2f}.chpn", mode="min"):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.mode = ReduceMode(mode)
        self.best = float("inf") if mode == ReduceMode.MIN else -float("inf")

    def on_train_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self):
        # TODO zakirov(1.11.19) Add support for saving based on metric
        current = self.state.val_loss.avg or self.state.train_loss.avg
        if (self.mode == ReduceMode.MIN and current < self.best) or (
            self.mode == ReduceMode.MAX and current > self.best
        ):  
            ep = self.state.epoch
            save_name = os.path.join(self.save_dir, self.save_name.format(ep=ep, metric=current))
            self._save_checkpoint(save_name)

    def _save_checkpoint(self, path):
        if hasattr(self.state.model, "module"):  # used for saving DDP models
            state_dict = self.state.model.module.state_dict()
        else:
            state_dict = self.state.model.state_dict()
        torch.save(
            {
                "epoch": self.state.epoch,
                "state_dict": state_dict,
                "optimizer": self.state.optimizer.state_dict(),
            },
            path,
        )


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

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_batch_end(self):
        # TODO: somehow account for different batch size
        self.current_step += 1
        if self.state.is_train and (self.current_step % self.log_every == 0):
            self.writer.add_scalar(
                # need proper name
                "train_/loss", self.state.loss_meter.val, self.current_step
            )
            for m in self.state.metric_meters:
                self.writer.add_scalar("train_/{}".format(m.name), m.val, self.current_step)

    def on_epoch_end(self):
        self.writer.add_scalar(
            "train/loss", self.state.train_loss.avg, self.current_step
        )
        for m in self.state.train_metrics:
            self.writer.add_scalar("train/{}".format(m.name), m.avg, self.current_step)

        lr = sorted([pg["lr"] for pg in self.state.optimizer.param_groups])[-1]  # largest lr
        # why train_ ? 
        self.writer.add_scalar("train_/lr", lr, self.current_step)

        # don't log if no val
        if self.state.val_loss is None:
            return

        self.writer.add_scalar("val/loss", self.state.val_loss.avg, self.current_step)
        for m in self.state.val_metrics:
            self.writer.add_scalar("val/{}".format(m.name), m.avg, self.current_step)

    def on_train_end(self):
        self.writer.close()


class Logger(Callback):
    def __init__(self, log_dir, logger=None):
        # logger - already created instance of logger
        super().__init__()
        self.logger = logger or self._get_logger(os.path.join(log_dir, "logs.txt"))

    def on_epoch_begin(self):
        self.logger.info(f"Epoch {self.state.epoch} | lr {self.current_lr:.3f}")

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
