
from ..utils.misc import AverageMeter
from ..utils.misc import TimeMeter
from ..utils.misc import to_numpy
import torch
from tqdm import tqdm
from torch import nn
from apex import amp


def get_val(metrics):
    results = [(m.name, m.avg()) for m in metrics]
    names, vals = list(zip(*results))
    out = ['{} : {:4f}'.format(name, val) for name, val in results]
    return vals, ' | '.join(out)


class FitWrapper(nn.Module):
    def __init__(self, model):
        super(FitWrapper, self).__init__()
        self.model = model

    def compile(self, opt, criterion, metrics):
        self.opt = optimizer
        self.criterion = criterion
        self.metrics = metrics

    def _fit_epoch(generator, steps_per_epoch=None):
        timer = TimeMeter()
        loss_meter = AverageMeter()
        metric_meters [AverageMeter() for i in self.metrics]
        model.train()
        pbar = tqdm(generator, ascii=True)
        for batch, target in pbar:
            timer.batch_start()
            #TODO add sheduler
            # compute output
            output = self.model(batch)
            loss = self.criterion(model)
            # compute grads
            self.opt.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            # sync
            torch.cuda.synchronize()
            timer.batch_end()
            bs = batch.size(0)
            loss_meter.update(loss)
            out = ['{} : {:4f}'.format(m.name, mm.avg_smooth) for m, mm in zip(self.metrics, metric_meters)]
            desc = 'Loss: {}'.format(loss_meter.avg_smooth) + ' | '.join(out)
            pbar.set_description(desc)
        return loss_meter.avg # + return something else

    def fit_generator(generator, 
                      steps_per_epoch=None, # not yet supported
                      epochs=1, 
                      val_generator=None, 
                      val_steps=None, 
                      val_freq=1, 
                      #initial_epoch=0 #not used
                      )

        # do something
        for ep in range(epochs):
            self._fit_epoch(generator)
            
            if val_generator and ep % val_freq:
                val_loss, val_m  = self.evaluate_generator(val_generator, val_steps)
                    

    #TODO add predict_generator

    def evaluate_generator(self, generator, *, steps=None, return_pred=False):
        timer = TimeMeter()
        loss_meter = AverageMeter()
        metric_meters [AverageMeter() for i in self.metrics]
        model.eval()
        preds_list = []
        pbar = tqdm(generator, ascii=True)
        for batch, target in pbar:
            timer.batch_start()
            with torch.no_grad():
                output = self.model(batch)
                loss = self.criterion(output, target).data
                for metric, meter in zip(self.metrics, metric_meters):
                    meter.update(metric(output, target))
            bs = batch.size(0)
            loss_meter.update(loss)
            if return_pred:
                preds_list.append(to_numpy(output))
            torch.cuda.synchronize()
            timer.batch_end()
            # TODO add info from timer
            out = ['{} : {:4f}'.format(m.name, mm.avg_smooth) for m, mm in zip(self.metrics, metric_meters)]
            desc = 'Loss: {}'.format(loss_meter.avg_smooth) + ' | '.join(out)
            pbar.set_description(desc)
        if return_pred:
            return loss_meter.avg, [m.avg for m in metric_meters], preds_list
        else:
            return loss_meter.avg, [m.avg for m in metric_meters]