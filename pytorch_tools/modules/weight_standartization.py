from torch import nn
import torch.nn.functional as F

# implements idea from `Weight Standardization` paper https://arxiv.org/abs/1903.10520
# eps is inside sqrt to avoid overflow Idea from https://arxiv.org/abs/1911.05920    
class WS_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        weight = self.weight
        weight = weight.sub(weight.mean(dim=(1, 2, 3), keepdim=True))
        std = weight.var(dim=(1, 2, 3), keepdim=True).add_(1e-7).sqrt_()
        weight = weight.div(std.expand_as(weight))
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# code from random issue on github. 
def convertConv2WeightStand(module, nextChild=None):
    mod = module
    norm_list = [torch.nn.modules.batchnorm.BatchNorm1d, torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.BatchNorm3d, torch.nn.GroupNorm, torch.nn.LayerNorm]
    conv_list = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d]
    for norm in norm_list:
        for conv in conv_list:
            if isinstance(mod, conv) and isinstance(nextChild, norm):
                mod = Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, mod.stride,
                 mod.padding, mod.dilation, mod.groups, mod.bias!=None)

    moduleChildList = list(module.named_children())
    for index, [name, child] in enumerate(moduleChildList):
        nextChild = None
        if index < len(moduleChildList) -1:
            nextChild = moduleChildList[index+1][1]
        mod.add_module(name, convertConv2WeightStand(child, nextChild))

    return mod

