
from ..utils.misc import AverageMeter
from ..utils.misc import TimeMeter
from ..utils.misc import to_numpy
from ..utils.misc import listify
import torch
#from tqdm.autonotebook import tqdm
from tqdm.auto import tqdm
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
        # TODO move amp logic here
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = listify(metrics)

    def _fit_epoch(self, generator, steps_per_epoch=None):
        timer = TimeMeter()
        loss_meter = AverageMeter()
        metric_meters = [AverageMeter() for i in self.metrics]
        self.model.train()
        steps = steps_per_epoch or len(generator)
        pbar = tqdm(generator, leave=False, total=steps)
        for idx, batch in enumerate(pbar):
            if idx == steps_per_epoch:
                break
            data, target = batch
            timer.batch_start()
            #TODO add sheduler
            # compute output
            output = self.model(data)
            loss = self.criterion(output, target)
            # compute grads
            self.optimizer.zero_grad()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()
            # calc metrics
            for metric, meter in zip(self.metrics, metric_meters):
                meter.update(to_numpy(metric(output, target)).squeeze())
            loss_meter.update(to_numpy(loss).squeeze())
            # sync
            torch.cuda.synchronize()
            timer.batch_end()
            bs = batch.size(0)
            out = self._format_meters(metric_meters)
            desc = 'Loss: {:4.4f} |'.format(loss_meter.avg_smooth) + out
            pbar.set_description(desc)
        out = self._format_meters(metric_meters, smooth=False)
        desc = 'Loss: {:4.4f} |'.format(loss_meter.avg) + out
        return desc

    def fit_generator(self, 
                      generator, 
                      steps_per_epoch=None,
                      epochs=1, 
                      val_generator=None, 
                      val_steps=None, 
                      val_freq=1, 
                      initial_epoch=0
                      ):

        # do something
        for ep in range(initial_epoch, epochs):
            # callbacks.on_epoch_begin
            desc = self._fit_epoch(generator)
            if (val_generator is not None) and (ep % val_freq == 0):
                val_loss, val_m  = self.evaluate_generator(val_generator, val_steps, use_pbar=False)
                val_desc = '| Val_loss: {:4.4f}'.format(val_loss)
                desc += val_desc
            print('Ep {:2d}/{:2d} |'.format(ep, epochs) + desc)
            # callbacks.on_epoch_end
        # callbacks.on_train_end

    #TODO add predict_generators

    def evaluate_generator(self, generator, steps=None, *, use_pbar=True):
        # eval metrics on generator
        timer = TimeMeter()
        loss_meter = AverageMeter()
        metric_meters = [AverageMeter() for i in self.metrics]
        self.model.eval()
        steps = steps or len(generator)
        pbar = tqdm(generator, leave=False, total=steps) if use_pbar else generator
        for idx, batch in enumerate(pbar):
            if idx == steps_per_epoch:
                break
            data, target = batch
            timer.batch_start()
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target).data 
            loss_meter.update(to_numpy(loss).squeeze()) #shape is (1,)
            for metric, meter in zip(self.metrics, metric_meters):
                meter.update(to_numpy(metric(output, target).squeeze()))
            #bs = batch.size(0)
            torch.cuda.synchronize()
            timer.batch_end()
            # TODO add info from timer
            out = self._format_meters(metric_meters)
            desc = 'Loss: {:4.4f} |'.format(loss_meter.avg_smooth) + out
            if use_pbar:
                pbar.set_description(desc)
        return loss_meter.avg, [m.avg for m in metric_meters]

    def _format_meters(self, meters, smooth=True):
        # meters: instances of AverageMeter
        results = [(m.name, meter.avg_smooth if smooth else meter.avg) 
                    for m, meter in zip(self.metrics, meters)]
        out = ['{}: {:6.3f}'.format(name, val) for name, val in results]
        return ' | '.join(out)
    
    def forward(self, x):
        return self.model.forward(x)