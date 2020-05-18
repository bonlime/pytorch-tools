import os
import math
import logging
from tqdm import tqdm
from enum import Enum
from collections import OrderedDict
from collections import defaultdict
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from .state import RunnerStateGAN
from pytorch_tools.fit_wrapper.callbacks import *
from pytorch_tools.utils.misc import listify


class CallbackGAN(Callback):
    def __init__(self):
        self.state = RunnerStateGAN()


class CallbacksGAN(CallbackGAN, Callbacks):
    """Class that combines multiple callbacks into one. For internal use only"""

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = listify(callbacks)


class ConsoleLoggerGAN(CallbackGAN, ConsoleLogger):
    def on_loader_end(self):
        # update to avg
        desc = OrderedDict(
            {"Gen Loss": f"{self.state.loss_meter.avg:.4f}",
             "Disc Loss": f"{self.state.loss_meter_disc.avg:.4f}"})
        desc.update({m.name: f"{m.avg:.3f}" for m in self.state.metric_meters})
        self.pbar.set_postfix(**desc)
        self.pbar.update()
        self.pbar.close()

    def on_batch_end(self):
        desc = OrderedDict(
            {"Gen loss": f"{self.state.loss_meter.avg_smooth:.4f}",
             "Disc loss": f"{self.state.loss_meter_disc.avg_smooth:.4f}",})
        desc.update({m.name: f"{m.avg_smooth:.3f}" for m in self.state.metric_meters})
        self.pbar.set_postfix(**desc)
        self.pbar.update()

class TensorBoardGAN(CallbackGAN, TensorBoard):
    def __init__(self, log_dir, log_every=20):
        TensorBoard.__init__(self, log_dir, log_every)
        # Noise to check progress of generator
        self.first_input = None

    def on_batch_end(self):
        # Save first val batch
        if not self.state.is_train and self.first_input is None:
            self.first_input = self.state.input

        self.current_step += self.state.batch_size
        if self.state.is_train and (self.current_step % self.log_every == 0):
            self.writer.add_scalar(
                # need proper name
                "train_/loss_gen",
                self.state.loss_meter.val,
                self.current_step,
            )
            self.writer.add_scalar(
                # need proper name
                "train_/loss_disc",
                self.state.loss_meter.val,
                self.current_step,
            )
            for m in self.state.metric_meters:
                self.writer.add_scalar(f"train_/{m.name}", m.val, self.current_step)

    def on_epoch_end(self):
        # Log scalars
        self.writer.add_scalar("train/loss_gen", self.state.train_loss.avg, self.current_step)
        self.writer.add_scalar("train/loss_disc", self.state.train_loss_disc.avg, self.current_step)
        for m in self.state.train_metrics:
            self.writer.add_scalar(f"train/{m.name}", m.avg, self.current_step)
        lr_gen = sorted([pg["lr"] for pg in self.state.optimizer.param_groups])[-1]  # largest lr
        lr_disc = sorted([pg["lr"] for pg in self.state.optimizer_disc.param_groups])[-1]  # largest lr
        self.writer.add_scalar("train_/lr_gen", lr_gen, self.current_step)
        self.writer.add_scalar("train_/lr_disc", lr_disc, self.current_step)
        self.writer.add_scalar("train/epoch", self.state.epoch, self.current_step)

        # don't log if no val
        if self.state.val_loss is None:
            return

        # Log scalars
        self.writer.add_scalar("val/loss_gen", self.state.val_loss.avg, self.current_step)
        self.writer.add_scalar("val/loss_disc", self.state.val_loss_disc.avg, self.current_step)
        for m in self.state.val_metrics:
            self.writer.add_scalar(f"val/{m.name}", m.avg, self.current_step)

        # Log images
        N = 16
        output = self.state.model(self.first_input[0])
        grid_target = torchvision.utils.make_grid(self.first_input[1][:N], nrow=int(math.sqrt(N)), normalize=True)
        grid_output = torchvision.utils.make_grid(output[:N], nrow=int(math.sqrt(N)), normalize=True)
        # Concat along X axis
        final_image = torch.cat([grid_output, grid_target], dim=2)
        self.writer.add_image(f'Images', final_image, self.state.epoch)


class PhasesSchedulerGAN(CallbackGAN, PhasesScheduler):
    """
    Scheduler that uses `phases` to process updates.

    Args:
        phases (List[Dict]): phases
        change_every (int): how often to actually change the lr. changing too 
            often may slowdown the training
        mode (str): Which optimizer to update {"gen", "disc", "all"}
        

    Example:
        PHASES = [
            {"ep":[0,8],  "lr":[0,0.1], "mom":0.9, },
            {"ep":[8,24], "lr":[0.1, 0.01], "mode":"cos"},
            {'ep':[24, 30], "lr": 0.001},
        ]
    """

    def __init__(self, phases, change_every=50, mode="all"):
        self.change_every = change_every
        self.current_lr = None
        self.current_mom = None
        self.phases = [self._format_phase(p) for p in phases]
        self.phase = self.phases[0]
        self.tot_epochs = max([max(p["ep"]) for p in self.phases])
        self.mode = mode
        super().__init__()


    def on_batch_begin(self):
        lr, mom = self._get_lr_mom(self.state.step)
        if (self.current_lr == lr and self.current_mom == mom) or (self.state.step % self.change_every != 0):
            return
        self.current_lr = lr
        self.current_mom = mom
        if self.mode == "gen":
            for param_group in self.state.optimizer.param_groups:
                param_group["lr"] = lr
                param_group["momentum"] = mom
        elif self.mode == "disc":
            for param_group in self.state.optimizer_disc.param_groups:
                param_group["lr"] = lr
                param_group["momentum"] = mom
        elif self.mode == "all":
            for param_group in self.state.optimizer.param_groups:
                param_group["lr"] = lr
                param_group["momentum"] = mom

            for param_group in self.state.optimizer_disc.param_groups:
                param_group["lr"] = lr
                param_group["momentum"] = mom