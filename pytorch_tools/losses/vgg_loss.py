"""
Implementation of VGG16 loss, originaly used for style transfer and usefull in many other task (including GAN training)
"""
from pytorch_tools import models

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

class ContentLoss(_Loss):
    """
    Creates content loss for neural style transfer
    model: str in ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
    """
    models_list = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

    def __init__(self, model="vgg16_bn", **args):
        super().__init__()
        try:
            vgg = models.__dict__[arch](pretrained=pretrained, **args)
        except KeyError:
            print("Model architecture not found in {}".format(models_list))
        vgg
   

    def forward(self, content, style):
        pass

class StyleLoss(_Loss):
    """
    Class for creating style loss for neural style transfer
    model: str in ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
    """
    models_list = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

    def __init__(self, model="vgg16_bn", **args):
        super().__init__()
        try:
            vgg = models.__dict__[arch](pretrained=pretrained, **args)
        except KeyError:
            print("Model architecture not found in {}".format(models_list))
        features = vgg
    
    @staticmethod
    def gram_matrix(input, normalize=True):
        """
        Calculate Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        input: Tensor with shape BxCxHxW
               B: batch shape (==1), C: number of channels, H&W: spatial dimmensions
        """
        B, C, H, W = input.size()
        assert B == 1, "Batch size should be 1"
        features = input.view(C, H * W)
        
        # Compute Gramm product
        gramm = torch.mm(features, features.t()) 

        if normalize:
            gramm.div(C * H * W)
        return gramm

    def forward(self, content, style):
        pass


