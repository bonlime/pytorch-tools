# code based on https://github.com/michaelrzhang/lookahead/
# but with unnesesary functions removed

import torch
from torch.optim import Optimizer
from collections import defaultdict


class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, base_optimizer, la_alpha=0.5, la_steps=5):
        """
        Args:
            base_optimizer : inner optimizer
            la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer
            la_steps (int): number of lookahead steps.
        """
        if not 0.0 <= la_alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {la_alpha}")
        if not 1 <= la_steps:
            raise ValueError(f"Invalid lookahead steps: {la_steps}")
        self.optimizer = base_optimizer
        self.la_alpha = la_alpha
        self.la_steps = la_steps
        self._total_steps = 0

        self.state = defaultdict(dict)
        # Cache the current optimizer parameters
        for group in base_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["slow_params"] = torch.zeros_like(p.data)
                param_state["slow_params"].copy_(p.data)  # disables grad

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._total_steps += 1
        if self._total_steps % self.la_steps == 0:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    # φ = φ + α * (θ - φ) = α * θ + (1 - α) * φ
                    p.data.mul_(self.la_alpha).add_(1.0 - self.la_alpha, param_state["slow_params"])
                    param_state["slow_params"].copy_(p.data)
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    # using property to have the same reference after load
    @property
    def param_groups(self):
        return self.optimizer.param_groups
