
from copy import copy
from enum import Enum
from collections import OrderedDict
from tqdm.auto import tqdm
import torch
from apex import amp
from ..utils.misc import AverageMeter
from ..utils.misc import TimeMeter
from ..utils.misc import to_numpy
from ..utils.misc import listify
from .callbacks import Callbacks


class Runner:
    def __init__(self, model, optimizer, criterion, metrics=None, callbacks=None, verbose=True):
        super(Runner, self).__init__()
        # TODO move amp logic here
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_runner(self)
        self._metrics = listify(metrics)
        self._metric_meters = [AverageMeter(name=m.name) for m in self._metrics]
        self._loss_meter = AverageMeter('loss')
        self._timer = TimeMeter()
        self._epochs = 1
        self._epoch = 1
        self._verbose = verbose
        self._train_metrics = None
        self._val_metrics = None
        self._is_train = None
        self._ep_size = None
        self._step = None

    def fit(self,
            train_loader,
            steps_per_epoch=None,
            val_loader=None,
            val_steps=None,
            epochs=2,
            start_epoch=1):

        self._epochs = epochs
        self.callbacks.on_train_begin()
        for epoch in range(start_epoch, epochs+1):
            self._is_train = True  # added to always know the state
            self._epoch = epoch
            self.callbacks.on_epoch_begin()
            self.model.train()
            self._run_one_epoch(train_loader, steps=steps_per_epoch)
            self._train_metrics = copy(self._loss_meter), [copy(m) for m in self._metric_meters]

            if val_loader is not None:
                self.evaluate(val_loader, steps=val_steps)
                self._val_metrics = copy(self._loss_meter), [copy(m) for m in self._metric_meters]

            self.callbacks.on_epoch_end()
        self.callbacks.on_train_end()

    def evaluate(self, loader, steps=None):
        self._is_train = False
        self.model.eval()
        self._run_one_epoch(loader, steps=steps)
        return self._loss_meter.avg, [m.avg for m in self._metric_meters]

    def _make_step(self, batch):
        images, target = batch
        output = self.model(images)
        loss = self.criterion(output, target)
        if self._is_train:
            self.optimizer.zero_grad()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            #grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            torch.cuda.synchronize()

        # update metrics
        self._loss_meter.update(to_numpy(loss))
        for metric, meter in zip(self._metrics, self._metric_meters):
            meter.update(to_numpy(metric(output, target).squeeze()))

    def _run_one_epoch(self, loader, steps=None):
        self._loss_meter.reset()
        self._timer.reset()
        for metric in self._metric_meters:
            metric.reset()
        self._ep_size = len(loader)  # useful in callbacks
        if self._verbose:
            pbar = tqdm(enumerate(loader), total=steps or self._ep_size, ncols=0)
            pbar.set_description("Epoch {:2d}/{}. {}ing".format(
                self._epoch, self._epochs, ['validat', 'train'][self._is_train]))
        else:
            pbar = enumerate(loader)

        with torch.set_grad_enabled(self._is_train):
            for i, batch in pbar:
                if steps and i == steps:
                    if self._verbose:
                        pbar.close()
                    break
                self._step = i
                self.callbacks.on_batch_begin()
                self._make_step(batch)
                if self._verbose:
                    desc = OrderedDict({'Loss': "{:.4f}".format(self._loss_meter.avg_smooth)})
                    desc.update({m.name: "{:.3f}".format(m.avg_smooth) for m in self._metric_meters})
                    pbar.set_postfix(**desc)
                self.callbacks.on_batch_end()
        return
