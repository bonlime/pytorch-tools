import torch
from torch.optim.optimizer import Optimizer, required

# NOTE The only diffrence between this and native torch.optim.SGD is at line 86
class SGDW(Optimizer):
    r"""Implements SGDW algorithm.
    The SGDW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
       params (iterable): iterable of parameters to optimize or dicts defining
           parameter groups
       lr (float): learning rate
       momentum (float, optional): momentum factor (default: 0)
       weight_decay (float, optional): weight decay coefficient (default: 1e-2)
       dampening (float, optional): dampening for momentum (default: 0)
       nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
       >>> optimizer = torch.optim.SGDW(model.parameters(), lr=0.1, momentum=0.9)
       >>> optimizer.zero_grad()
       >>> loss_fn(model(input), target).backward()
       >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
       The implementation of SGD with Momentum/Nesterov subtly differs from
       Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                 v = \rho * v + g \\
                 p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
       velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
       other frameworks which employ an update of the form
        .. math::
            v = \rho * v + lr * g \\
            p = p - v
        The Nesterov version is analogously modified.
    .. _Decoupled Weight Decay Regularization:
       https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
           closure (callable, optional): A closure that reevaluates the model
               and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # Apply weight decay. THE ONLY DIFFERENCE IS HERE
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                # Apply momentum
                p.data.add_(-group["lr"], d_p)
        return loss
