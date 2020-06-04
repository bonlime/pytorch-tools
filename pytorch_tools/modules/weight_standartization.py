import torch
from torch import nn
import torch.nn.functional as F

# implements idea from `Weight Standardization` paper https://arxiv.org/abs/1903.10520
# eps is inside sqrt to avoid overflow Idea from https://arxiv.org/abs/1911.05920
class WS_Conv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        var, mean = torch.var_mean(weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (weight - mean) / torch.sqrt(var + 1e-7)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# code from SyncBatchNorm in pytorch
def conv_to_ws_conv(module):
    module_output = module
    if isinstance(module, torch.nn.Conv2d):
        module_output = WS_Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            # groups are also present in DepthWiseConvs which we don't want to patch
            # TODO: fix this
            groups=module.groups,
            bias=module.bias is not None,
        )
        with torch.no_grad():  # not sure if torch.no_grad is needed. but just in case
            module_output.weight.copy_(module.weight)
            module_output.weight.requires_grad = module.weight.requires_grad
            if module.bias is not None:
                module_output.bias.copy_(module.bias)
                module_output.bias.requires_grad = module.bias.requires_grad

    for name, child in module.named_children():
        module_output.add_module(name, conv_to_ws_conv(child))
    del module
    return module_output
