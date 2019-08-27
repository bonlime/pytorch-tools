
from ..utils.misc import AverageMeter
from ..utils.misc import TimeMeter
from ..utils.misc import to_numpy
from ..utils.misc import listify
from collections import OrderedDict
import torch
from tqdm.auto import tqdm
from torch import nn
from apex import amp
from .callbacks import Callbacks
from copy import copy


class Runner:
    def __init__(self, model):
        super(Runner, self).__init__()
        self.model = model

    def compile(self, optimizer, criterion, metrics=None, callbacks=None):
        # TODO move amp logic here
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = listify(metrics)
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_runner(self)
        self.metric_meters = [AverageMeter(name=m.name) for m in self.metrics]
        self.loss_meter = AverageMeter('loss')
        self.timer = TimeMeter()

    def fit(self, 
            train_loader, 
            steps_per_epoch=None,
            val_loader=None,
            val_steps=None, 
            epochs=1, 
            start_epoch=0):
    
        self.n_epoch = epochs
        self.callbacks.on_train_begin()
        for epoch in range(start_epoch, epochs):
            self.callbacks.on_epoch_begin(epoch)
            self.epoch = epoch
            self.model.train()
            self._run_one_epoch(train_loader, is_train=True, steps=steps_per_epoch)
            
            if val_loader is not None:
                # useful in callbacks
                self._train_metrics = self.loss_meter.avg, [copy(m) for m in self.metric_meters]
                self.evaluate(val_loader, steps=val_steps)

            self.callbacks.on_epoch_end(epoch)
        self.callbacks.on_train_end()

    #TODO add predict_generators
    def evaluate(self, loader, steps=None):
        if not hasattr(self, 'n_epoch'):
            self.n_epoch = 1
            self.epoch = 1
        self.model.eval()
        self._run_one_epoch(loader, is_train=False, steps=steps)
        return self.loss_meter.avg, [m.avg for m in self.metric_meters]

    def _make_step(self, batch, is_train):
        images, target = batch
        output = self.model(images)
        loss = self.criterion(output, target)
        if is_train:
            self.optimizer.zero_grad()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            #grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            torch.cuda.synchronize()
            
        # update metrics
        self.loss_meter.update(to_numpy(loss))
        for metric, meter in zip(self.metrics, self.metric_meters):
            meter.update(to_numpy(metric(output, target).squeeze()))

        return None # or smth? 

    def _run_one_epoch(self, loader, is_train=True, steps=None):
        self.loss_meter.reset()
        self.timer.reset()
        for m in self.metric_meters:
            m.reset()
        self.ep_size = len(loader) # useful in callbacks
        pbar = tqdm(enumerate(loader), total=steps or self.ep_size, ncols=0) #, ncols=0
        pbar.set_description("Epoch {:2d}/{}. {}ing:".format(
            self.epoch, self.n_epoch, ['validat', 'train'][is_train]))
        with torch.set_grad_enabled(is_train):
            for i, batch in pbar:
                if steps and i == steps:
                    pbar.close()
                    break
                self.callbacks.on_batch_begin(i)
                self._make_step(batch, is_train)
                desc = OrderedDict({'Loss': "{:.4f}".format(self.loss_meter.avg_smooth)})
                desc.update({m.name: "{:.3f}".format(m.avg_smooth) for m in self.metric_meters})
                pbar.set_postfix(**desc)
                self.callbacks.on_batch_end(i)
        return None
