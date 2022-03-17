"""To test importing modules from another file"""
import torch
from typing import List


class Concat(torch.nn.Module):
    def forward(self, tensors: List[torch.Tensor]):
        return torch.cat(tensors, dim=1)
