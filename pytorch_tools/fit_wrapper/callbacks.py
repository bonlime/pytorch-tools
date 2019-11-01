import os
import math
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from ..utils.misc import listify


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

    def set_runner(self, runner):
        self.runner = runner

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
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

    def set_runner(self, runner):
        super().set_runner(runner)
        for callback in self.callbacks:
            callback.set_runner(runner)

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


class PhasesScheduler(Callback):
    """
    LR and momentum scheduler that uses `phases` to prosses
    updates.
    Example:
    LOADED_PHASES = [
        {'ep':(0,8),  'lr':(lr/2,lr*2), 'mom':0.9, 'mode':'cos'}, # lr warmup is better with --init-bn0
        {'ep':(8,24), 'lr':(lr*2,lr/2), 'mode':'cos'}, # trying one cycle
        {'ep':(24, 30), 'lr':(lr*bs_scale[1], lr/5*bs_scale[1])},
        {'ep':(30, 33), 'lr':(lr/5*bs_scale[2], lr/25*bs_scale[2])},
        {'ep':(33, 34), 'lr':(lr/25*bs_scale[2], lr/125*bs_scale[2])},
    ]
    """
    def __init__(self, optimizer, phases, change_every=10):
        self.optimizer = optimizer
        self.change_every = change_every
        self.current_lr = None
        self.current_mom = None
        self.phases = [self._format_phase(p) for p in phases]
        self.phase = self.phases[0]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])
        super(PhasesScheduler, self).__init__()

    def _format_phase(self, phase):
        phase['ep'] = listify(phase['ep'])
        phase['lr'] = listify(phase['lr'])
        phase['mom'] = listify(phase.get('mom', None)) # optional
        if len(phase['lr']) == 2 or len(phase['mom']) == 2:
            phase['mode'] = phase.get('mode', 'linear') 
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    @staticmethod
    def _schedule(start, end, pct, mode):
        """anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        if mode == 'linear':
            return start + (end - start) * pct
        elif mode == 'cos':
            return end + (start - end)/2 * (math.cos(math.pi * pct) + 1)

    def _get_lr_mom(self, batch_curr):
        phase = self.phase
        batch_tot = self.runner._ep_size
        if len(phase['ep']) == 1:
            perc = 0
        else:
            ep_start, ep_end = phase['ep']
            ep_curr, ep_tot = self.runner._epoch - ep_start, ep_end - ep_start
            perc = (ep_curr * batch_tot + batch_curr) / (ep_tot * batch_tot)
        if len(phase['lr']) == 1:
            new_lr = phase['lr'][0] # constant learning rate
        else:
            lr_start, lr_end = phase['lr']
            new_lr = self._schedule(lr_start, lr_end, perc, phase['mode'])
            
        if len(phase['mom']) == 0:
            new_mom = self.current_mom
        elif len(phase['mom']) == 1:
            new_mom = phase['mom'][0]
        else:
            mom_start, mom_end = phase['mom']
            new_mom = self._schedule(mom_start, mom_end, perc, phase['mode'])

        return new_lr, new_mom

    def on_epoch_begin(self):
        new_phase = None
        for phase in reversed(self.phases):
            if (self.runner._epoch >= phase['ep'][0]):
                new_phase = phase
                break
        if new_phase is None:
            raise Exception('Epoch out of range')
        else:
            self.phase = new_phase

    def on_batch_begin(self):
        lr, mom = self._get_lr_mom(self.runner._step)
        if (self.current_lr == lr and self.current_mom == mom) or (self.runner._step % self.change_every != 0):
            return
        self.current_lr = lr
        self.current_mom = mom
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = mom

class CheckpointSaver(Callback):
    def __init__(self, save_dir, save_name='model_{ep}_{metric:.2f}.chpn', mode='min'):
        super().__init__()
        self.mode = mode
        self.save_dir = save_dir
        self.save_name = save_name
        self.best = float('inf') if mode == 'min' else -float('inf')

    def on_train_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self):
        # TODO zakirov(1.11.19) Add support for saving based on metric
        if self.runner._val_metrics is not None:
            metric = self.runner._val_metrics[0].avg # loss
        else:
            metric = self.runner._train_metrics[0].avg
        if (self.mode == 'min' and metric < self.best) or \
           (self.mode == 'max' and metric > self.best):
            save_name = os.path.join(
                self.save_dir, self.save_name.format(ep=self.runner._epoch, metric=metric))
            self._save_checkpoint(save_name)

    def _save_checkpoint(self, path):
        torch.save({
            'epoch': self.runner._epoch,
            'state_dict': self.runner.model.state_dict(),
            'optimizer': self.runner.optimizer.state_dict()}, path)


#     def on_epoch_end(self, epoch):
        
#         metric = self.metrics.val_metrics[self._metric_name]
#         new_path_to_save = os.path.join(
#             self.save_dir / self.runner.current_stage_name,
#             self.save_name.format(epoch=epoch, metric="{:.5}".format(metric)))
#         if self._try_update_best_losses(metric, new_path_to_save):
#             self.save_checkpoint(epoch=epoch, path=new_path_to_save)

#     def _try_update_best_losses(self, metric, new_path_to_save):
#         if self.mode == 'min':
#             metric = -metric
#         if not self._best_checkpoints_queue.full():
#             self._best_checkpoints_queue.put((metric, new_path_to_save))
#             return True

#         min_metric, min_metric_path = self._best_checkpoints_queue.get()

#         if min_metric < metric:
#             os.remove(min_metric_path)
#             self._best_checkpoints_queue.put((metric, new_path_to_save))
#             return True

#         self._best_checkpoints_queue.put((min_metric, min_metric_path))
#         return False


class TensorBoard(Callback):
    def __init__(self, log_dir, log_every=100):
        super().__init__()
        self.log_dir = log_dir
        self.log_every = log_every
        self.writer = None

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_batch_end(self):
        if self.runner._is_train and (self.runner._step % self.log_every == 0):
            self.writer.add_scalar('train_/loss', self.runner._loss_meter.val, self.global_step)
            for m in self.runner._metric_meters:
                self.writer.add_scalar('train_/{}'.format(m.name), m.val, self.global_step)

    def on_epoch_end(self):
        self.writer.add_scalar('train/loss', self.runner._train_metrics[0].avg, self.global_step)
        for m in self.runner._train_metrics[1]:
            self.writer.add_scalar('train/{}'.format(m.name), m.avg, self.global_step)

        self.writer.add_scalar('val/loss', self.runner._val_metrics[0].avg, self.global_step)
        for m in self.runner._val_metrics[1]:
            self.writer.add_scalar('val/{}'.format(m.name), m.avg, self.global_step)

        # for idx, param_group in enumerate(self.runner.optimizer.param_groups):
        #     lr = param_group['lr']
        #     self.writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=epoch)
    
    @property
    def global_step(self):
        return (self.runner._epoch - 1) * self.runner._ep_size + self.runner._step
    
    def on_train_end(self):
        self.writer.close()


class Logger(Callback):
    def __init__(self, log_dir, logger=None):
        # logger - already created instance of logger
        super().__init__()
        self.logger = logger or self._get_logger(os.path.join(log_dir, 'logs.txt'))

    def on_epoch_begin(self):
        self.logger.info(
            'Epoch {} | '.format(self.runner._epoch) + 'lr {:.3f}'.format(self.current_lr[0]))

    def on_epoch_end(self):
        self.logger.info('Train ' + self._format_meters(*self.runner._train_metrics))
        if self.runner._val_metrics is not None:
            self.logger.info('Val   ' + self._format_meters(*self.runner._val_metrics))

    @staticmethod
    def _get_logger(log_path):
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @property
    def current_lr(self):
        res = []
        for param_group in self.runner.optimizer.param_groups:
            res.append(param_group['lr'])
        return res

    @staticmethod
    def _format_meters(loss, metrics):
        return 'loss: {:.4f} | '.format(loss.avg) +  " | ".join("{}: {:.4f}".format(m.name, m.avg) for m in metrics)
