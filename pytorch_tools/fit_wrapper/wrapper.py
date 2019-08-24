
from ..utils.misc import AverageMeter
from ..utils.misc import TimeMeter
from ..utils.misc import to_numpy
from ..utils.misc import listify
from collections import OrderedDict
import torch
from tqdm.auto import tqdm
from torch import nn
from apex import amp


class FitWrapper:
    def __init__(self, model):
        super(FitWrapper, self).__init__()
        self.model = model

    def compile(self, optimizer, criterion, metrics=None):
        # TODO move amp logic here
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = listify(metrics)
        self.metric_meters = [AverageMeter(name=m.name) for m in self.metrics]
        self.loss_meter = AverageMeter('Loss')
        self.timer = TimeMeter()

    def fit(self, 
            train_loader, 
            epochs=1, 
            val_loader=None, 
            ):

        self.n_epoch = epochs
        for epoch in range(epochs):
            # callbacks.on_epoch_begin
            self.model.train()
            self._run_one_epoch(epoch, train_loader, is_train=True)

            #desc = self._fit_epoch(generator, steps_per_epoch)
            if val_loader is not None:
                self.model.eval()
                self._run_one_epoch(epoch, val_loader, is_train=False)
            # callbacks.on_epoch_end
        # callbacks.on_train_end

    #TODO add predict_generators
    def evaluate(self, loader):
        self.n_epoch = 1
        self.model.eval()
        self._run_one_epoch(1, loader, is_train=False)
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

    def _run_one_epoch(self, epoch, loader, is_train=True):
        self.loss_meter.reset()
        self.timer.reset()
        for m in self.metric_meters:
            m.reset()
        pbar = tqdm(enumerate(loader), total=len(loader)) #, ncols=0
        pbar.set_description("Epoch {:2d}/{}. {}ing:".format(
            epoch, self.n_epoch, ['validat', 'train'][is_train]))
        with torch.set_grad_enabled(is_train):
            for i, data in pbar:
                #self.callbacks.on_batch_begin(i)
                self._make_step(data, is_train)
                desc = OrderedDict({'Loss': "{:.4f}".format(self.loss_meter.avg_smooth)})
                desc.update({m.name: "{:.3f}".format(m.avg_smooth) for m in self.metric_meters})
                pbar.set_postfix(**desc)
                #self.callbacks.on_batch_end(i, step_report=step_report, is_train=is_train)
        return None
