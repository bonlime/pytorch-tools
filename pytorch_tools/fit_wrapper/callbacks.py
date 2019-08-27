import os
import torch
import logging
from queue import PriorityQueue
from tensorboardX import SummaryWriter
from ..utils.misc import listify
import math


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

    def on_batch_begin(self, i, **kwargs):
        pass

    def on_batch_end(self, i, **kwargs):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
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

    def on_batch_begin(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(i, **kwargs)

    def on_batch_end(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(i, **kwargs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

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
    def __init__(self, optimizer, phases):
        self.optimizer = optimizer
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
        batch_tot = self.runner.ep_size
        if len(phase['ep']) == 1:
            perc = 0
        else:
            ep_start, ep_end = phase['ep']
            ep_curr, ep_tot = self.epoch - ep_start, ep_end - ep_start
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

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        new_phase = None
        for phase in reversed(self.phases):
            if (epoch >= phase['ep'][0]):
                new_phase = phase
                break
        if new_phase is None:
            raise Exception('Epoch out of range')
        else:
            self.phase = new_phase

    def on_batch_begin(self, i):
        lr, mom = self._get_lr_mom(i)
        if self.current_lr == lr and self.current_mom == mom:
            return 
        self.current_lr = lr
        self.current_mom = mom
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = mom


# class CheckpointSaver(Callback):
#     def __init__(self, save_dir, save_name, mode='min', monitor='val_loss'):
#         super().__init__()
#         self.mode = mode
#         self.save_name = save_name
#         self._best_checkpoints_queue = PriorityQueue(5)
#         self.monitor_name = monitor
#         self.save_dir = save_dir

#     def on_train_begin(self):
#         os.makedirs(self.save_dir, exist_ok=True)
#         while not self._best_checkpoints_queue.empty():
#             self._best_checkpoints_queue.get()

#     def save_checkpoint(self, epoch, path):
#         if hasattr(self.runner.model, 'module'):
#             state_dict = self.runner.model.module.state_dict()
#         else:
#             state_dict = self.runner.model.state_dict()
#         torch.save({
#             'epoch': epoch + 1,
#             'state_dict': state_dict,
#             'optimizer': self.runner.optimizer.state_dict()}, path)

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


# class TensorBoard(Callback):
#     def __init__(self, log_dir: str):
#         super().__init__()
#         self.log_dir = log_dir
#         self.writer = None

#     def on_train_begin(self):
#         os.makedirs(self.log_dir, exist_ok=True)
#         self.writer = SummaryWriter(self.log_dir)

#     def on_epoch_end(self, epoch):
#         for k, v in self.metrics.train_metrics.items():
#             self.writer.add_scalar('train/{}'.format(k), float(v), global_step=epoch)

#         for k, v in self.metrics.val_metrics.items():
#             self.writer.add_scalar('val/{}'.format(k), float(v), global_step=epoch)

#         for idx, param_group in enumerate(self.runner.optimizer.param_groups):
#             lr = param_group['lr']
#             self.writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=epoch)

#     def on_train_end(self):
#         self.writer.close()


class Logger(Callback):
    def __init__(self, log_dir, logger=None):
        # logger - already created instance of logger
        super().__init__()
        self.logger = logger or self._get_logger(os.path.join(log_dir, 'logs.txt'))

    def on_epoch_begin(self, epoch):
        self.logger.info(
            'Epoch {} | '.format(epoch) + 'lr {:.3f}'.format(self.current_lr))

    def on_epoch_end(self, epoch):
        loss, metrics = self.runner.loss_meter.avg, self.runner.metric_meters
        has_val = bool(getattr(self.runner, '_train_metrics', None))
        trn_loss, trn_metrics = self.runner._train_metrics if has_val else (loss, metrics)
        self.logger.info('Train ' + self._format_meters(trn_loss, trn_metrics))
        if has_val:
            self.logger.info('Val ' + self._format_meters(loss, metrics))

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
        if len(res) == 1:
            return res[0]
        return res

    @staticmethod
    def _format_meters(loss, metrics):
        return 'loss: {:.4f} | '.format(loss) +  " | ".join("{}: {:.4f}".format(m.name, m.avg) for m in metrics)
