"""
Implementation of VGG16 loss, originaly used for style transfer and usefull in many other task (including GAN training)
It's work in progress, no guarantees that code will work
"""
from pytorch_tools import models
import torch.nn as nn
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
    layers: list of VGG layers used to evaluate content loss
    criterion: str in ['mse', 'mae'], reduction method
    """
    models_list = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

    def __init__(self, model="vgg19_bn", layers=["21"],
                 weights=1, loss="mse", device="cuda", **args):
        super().__init__()
        try:
            self.model = models.__dict__[arch](pretrained=True, **args)
            self.model.eval().to(device)
        except KeyError:
            print("Model architecture not found in {}".format(models_list))
        
        if isinstance(layers, str):
            layers = [layers]
        assert isinstance(layers, list), "Should be the list of weights or one str name"
        self.layers = layers

        if isinstance(weights, float) or isinstance(weights, int):
            weights = [weights]
        assert isinstance(weights, list), "Should be the list of weights or one number"
        self.weights = weights

        if criterion == "mse":
            self.criterion = nn.MSELoss()
        elif criterion == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise KeyError

    def forward(self, input, content):
        """
        Measure distance between feature representations of input and content images
        """
        input_features = get_features(input, self.model, self.layers)
        content_features = get_features(content, self.model, self.layers)
        loss = self.criterion(input_features, content_features)
        return loss

    def get_features(x, model, layers=None):
        """
        Extract features from the intermediate layers.
        """
        if layers is None:
            layers = ["21"]

        features = []
        for name, module in model.features._modules.items():
            x = module(x)
            if name in layers:
                features.append(x)
        return features

    
class StyleLoss(_Loss):
    """
    Class for creating style loss for neural style transfer
    model: str in ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
    """
    models_list = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

    def __init__(self, model="vgg19_bn", layers=["0", "5", "10", "19", "28"],
                 weights=[0.75, 0.5, 0.2, 0.2, 0.2], loss="mse", device="cuda", **args):
        super().__init__()
        try:
            self.model = models.__dict__[arch](pretrained=True, **args)
            self.model.eval().to(device)
        except KeyError:
            print("Model architecture not found in {}".format(models_list))
        
        if isinstance(layers, str):
            layers = [layers]
        assert isinstance(layers, list), "Should be the list of weights or one str name"
        self.layers = layers

        if isinstance(weights, float) or isinstance(weights, int):
            weights = [weights]
        assert isinstance(weights, list), "Should be the list of weights or one number"
        self.weights = weights

        if criterion == "mse":
            self.criterion = nn.MSELoss()
        elif criterion == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise KeyError

    def forward(self, input, style):
        """
        Measure distance between feature representations of input and content images
        """
        input_features = get_features(input, self.model, self.layers)
        style_features = get_features(content, self.model, self.layers)
        input_gram = gram_matrix(input_features)
        style_gram = gram_matrix(style_features)
        loss = self.criterion(input_gram, style_gram)
        return loss

    def get_features(x, model, layers=None):
        """
        Extract features from the intermediate layers.
        """
        if layers is None:
            layers = ["0", "5", "10", "19", "28"]

        features = []
        for name, module in model.features._modules.items():
            x = module(x)
            if name in layers:
                features.append(x)
        return features

    def gram_matrix(input):
        """
        Compute Gram matrix for each image in batch
        input: Tensor of shape BxCxHxW
            B: batch size
            C: channels size
            H&W: spatial size

        """

        B, C, H, W = input.size()
        gram = []
        for i in range(B):
            x = input[i].view(C, H * W)
            gram.append(torch.mm(x, x.t()))
        return gram