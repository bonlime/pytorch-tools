# it's a copy from torch.optim._multi_tensor with `eps` moved inside sqrt and `exp_avg_sq` init to ones
# Ref: Understanding the Disharmony between Weight Normalization Family and Weight Decay: ε−shifted L2 Regularizer
# the paper above basically says it's better for gradients to have epsilon inside sqrt
# Ref: On the Convergence of Adam and Beyond
# they also mention that it's possible instead to put higher epsilon inside sqrt instead of their fix
# the only change is on lines 136-142 and line 102
import math
import torch
from torch.optim import Optimizer
from collections import defaultdict


class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            amsgrad = group["amsgrad"]

            grads = []
            states = []
            exp_avg = []
            exp_avg_sq = []
            max_exp_avg_sq = []
            params_with_grad = []

            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")

                    # Perform stepweight decay
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                    params_with_grad.append(p)
                    grads.append(p.grad)

            for p in params_with_grad:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.ones_like(p, memory_format=torch.preserve_format)  # torch init to zeros
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg.append(state["exp_avg"])
                exp_avg_sq.append(state["exp_avg_sq"])

                if amsgrad:
                    max_exp_avg_sq.append(state["max_exp_avg_sq"])

                state["step"] += 1
                states.append(state)

            beta1, beta2 = group["betas"]

            bias_correction1 = [1 - beta1 ** state["step"] for state in states]
            bias_correction2 = [1 - beta2 ** state["step"] for state in states]

            #
            # Decay the first and second moment running average coefficient
            #
            torch._foreach_mul_(exp_avg, beta1)
            torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

            torch._foreach_mul_(exp_avg_sq, beta2)
            torch._foreach_addcmul_(exp_avg_sq, grads, grads, 1 - beta2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                max_exp_avg_sq = torch._foreach_maximum(max_exp_avg_sq, exp_avg_sq)

                # Use the max. for normalizing running avg. of gradient
                max_exp_avg_sq_sqrt = torch._foreach_sqrt(torch._foreach_add(max_exp_avg_sq, group["eps"]))
                bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
                denom = torch._foreach_div(max_exp_avg_sq_sqrt, bias_correction_sqrt)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(torch._foreach_add(exp_avg_sq, group["eps"]))
                bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
                denom = torch._foreach_div(exp_avg_sq_sqrt, bias_correction_sqrt)

            step_size = [-1 * (group["lr"] / bc) for bc in bias_correction1]
            torch._foreach_addcdiv_(params_with_grad, exp_avg, denom, step_size)

        return loss

    # TODO: refactor to a base class once foreach ops are in a good shape.
    def zero_grad(self, set_to_none: bool = False):
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)

                        if p.grad.is_sparse:
                            p.grad.zero_()
                        else:
                            per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)

            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._foreach_zero_(grads)
