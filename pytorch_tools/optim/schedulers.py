from torch.optim.lr_scheduler import _LRScheduler
from ..utils.misc import listify


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]



class PhasesScheduler():
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
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])

    def format_phase(self, phase):
        phase['ep'] = listify(phase['ep'])
        phase['lr'] = listify(phase['lr'])
        phase['mom'] = listify(phase.get('mom', None)) # optional
        if len(phase['lr']) == 2 or len(phase['mom']) == 2:
            phase['mode'] = phase.get('mode', 'linear') # optional 
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    def get_current_phase(self, epoch):
        for phase in reversed(self.phases):
            if (epoch >= phase['ep'][0]):
                return phase
        raise Exception('Epoch out of range')

    @staticmethod
    def _schedule(start, end, pct, mode):
        """anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        if mode == 'linear':
            return start + (end - start) * pct
        elif mode == 'cos':
            return end + (start - end)/2 * (math.cos(math.pi * pct) + 1)

    def get_lr_mom(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        if len(phase['ep']) == 1:
            perc = 0
        else:
            ep_start, ep_end = phase['ep']
            ep_curr, ep_tot = epoch - ep_start, ep_end - ep_start
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

    def update_lr_mom(self, epoch, batch_num, batch_tot):
        lr, mom = self.get_lr_mom(epoch, batch_num, batch_tot)
        if self.current_lr == lr and self.current_mom == mom:
            return

        self.current_lr = lr
        self.current_mom = mom
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = mom
            