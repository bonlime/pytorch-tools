
from ..utils.misc import AverageMeter
from ..utils.misc import TimeMeter
from ..utils.misc import to_numpy
import torch
from tqdm import tqdm
from torch import nn
from apex import amp



# fit_generator
#   _run_batch_train
# predict_generator
# evaluate_generator

# В керасе есть
# train on batch
# test on batch
# predict on batch
# когда надо сделать val, вызывает evalute_generator
# в eval считаем всякие лоссы. 
# в pred только прогоняем вход через сетку и стакаем, ничего не храним




class FitWrapper(nn.Module):
    def __init__(self, model):
        super(FitWrapper, self).__init__()
        self.model = model

    def compile(self, optimizer, criterion, metrics=None):
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics or [] #maybe need to listify

    def _fit_epoch(generator, steps_per_epoch=None):
        timer = TimeMeter()
        loss_meter = AverageMeter()
        metric_meters = [AverageMeter() for i in self.metrics]
        model.train()
        pbar = tqdm(generator, ascii=True)
        for batch, target in pbar:
            timer.batch_start()
            #TODO add sheduler
            # compute output
            output = self.model(batch)
            loss = self.criterion(model)
            # compute grads
            self.optimizer.zero_grad()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()
            # calc metrics
            for metric, meter in zip(self.metrics, metric_meters):
                    meter.update(to_numpy(metric(output, target)))
            loss_meter.update(to_numpy(loss))
            # sync
            torch.cuda.synchronize()
            timer.batch_end()
            bs = batch.size(0)
            out = self._format_meters(metric_meters)
            desc = 'Loss: {} |'.format(loss_meter.avg_smooth) + out
            pbar.set_description(desc)
        return loss_meter.avg, [m.avg for m in metric_meters]

    def fit_generator(generator, 
                      steps_per_epoch=None, # not yet supported
                      epochs=1, 
                      val_generator=None, 
                      val_steps=None, 
                      val_freq=1, 
                      #initial_epoch=0 #not used
                      ):

        # do something
        for ep in range(epochs):
            # callbacks.on_epoch_begin
            train_loss, train_m = self._fit_epoch(generator)
            if val_generator and ep % val_freq:
                val_loss, val_m  = self.evaluate_generator(val_generator, val_steps, use_pbar=False)
            # callbacks.on_epoch_end
        # callbacks.on_train_end

    #TODO add predict_generators

    def evaluate_generator(self, generator, steps=None, *, use_pbar=True):
        # eval metrics on generator
        timer = TimeMeter()
        loss_meter = AverageMeter()
        metric_meters = [AverageMeter() for i in self.metrics]
        model.eval()
        pbar = tqdm(generator, ascii=True) if use_pbar else generator
        for batch, target in pbar:
            timer.batch_start()
            with torch.no_grad():
                output = self.model(batch)
                loss = self.criterion(output, target).data #do we need data?
                loss_meter.update(loss)
                for metric, meter in zip(self.metrics, metric_meters):
                    meter.update(metric(output, target))
            #bs = batch.size(0)
            torch.cuda.synchronize()
            timer.batch_end()
            # TODO add info from timer
            out = self._format_meters(metric_meters)
            desc = 'Loss: {} |'.format(loss_meter.avg_smooth) + out
            if use_pbar:
                pbar.set_description(desc)
        
        return loss_meter.avg, [m.avg for m in metric_meters]

    def _format_meters(self, meters):
        # meters: instances of AverageMeter
        results = [(m.name, meter.avg_smooth) for m, meter in zip(self.metrics, meters)]
        out = ['{} : {:4f}'.format(name, val) for name, val in results]
        return ' | '.join(out)