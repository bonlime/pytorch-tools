"""To test importing modules from another file"""
import torch


class Concat(torch.nn.Module):
    def forward(self, *tensors):
        return torch.cat(tensors, dim=1)
