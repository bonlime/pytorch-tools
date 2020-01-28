"""
Implementation of VGG16 loss, originaly used for style transfer and usefull in many other task (including GAN training)
It's work in progress, no guarantees that code will work
"""
from pytorch_tools import models
import torch
import torch.nn as nn
from .base import Loss
from ..utils.misc import listify

MODELS_LIST = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]


class ContentLoss(Loss):
    """
    Creates content loss for neural style transfer
    model: str in ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
    layers: list of VGG layers used to evaluate content loss
    criterion: str in ['mse', 'mae'], reduction method
    """

    def __init__(
        self,
        model="vgg11_bn",
        pretrained="imagenet",
        layers=["21"],
        weights=1,
        loss="mse",
        device="cuda",
        **args,
    ):
        super().__init__()
        try:
            self.model = models.__dict__[model](pretrained=pretrained, **args)
            self.model.eval().to(device)
        except KeyError:
            print(f"Model architecture not found in {MODELS_LIST}")

        self.layers = listify(layers)
        self.weights = listify(weights)

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise KeyError

    def forward(self, input, content):
        """
        Measure distance between feature representations of input and content images
        """
        input_features = torch.stack(self.get_features(input))
        content_features = torch.stack(self.get_features(content))
        loss = self.criterion(input_features, content_features)

        # Solve big memory consumption
        torch.cuda.empty_cache()
        return loss

    def get_features(self, x):
        """
        Extract feature maps from the intermediate layers.
        """
        if self.layers is None:
            self.layers = ["21"]

        features = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        print(len(features))
        return features


class StyleLoss(Loss):
    """
    Class for creating style loss for neural style transfer
    model: str in ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
    """

    def __init__(
        self,
        model="vgg11_bn",
        pretrained="imagenet",
        layers=["0", "5", "10", "19", "28"],
        weights=[0.75, 0.5, 0.2, 0.2, 0.2],
        loss="mse",
        device="cuda",
        **args,
    ):
        super().__init__()
        try:
            self.model = models.__dict__[model](pretrained=pretrained, **args)
            self.model.eval().to(device)
        except KeyError:
            print(f"Model architecture not found in {MODELS_LIST}")

        self.layers = listify(layers)
        self.weights = listify(weights)

        # if isinstance(weights, float) or isinstance(weights, int):
        #     weights = [weights]
        # assert isinstance(weights, list), "Should be the list of weights or one number"
        # self.weights = weights

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise KeyError

    def forward(self, input, style):
        """
        Measure distance between feature representations of input and content images
        """
        input_features = self.get_features(input)
        style_features = self.get_features(style)
        print(style_features[0].size(), len(style_features))

        input_gram = [self.gram_matrix(x) for x in input_features]
        style_gram = [self.gram_matrix(x) for x in style_features]

        loss = 0
        # for i_g, s_g in zip(input_gram, style_gram):

        loss = [
            self.criterion(torch.stack(i_g), torch.stack(s_g)) for i_g, s_g in zip(input_gram, style_gram)
        ]
        return loss

    def get_features(self, x):
        """
        Extract feature maps from the intermediate layers.
        """
        if self.layers is None:
            self.layers = ["0", "5", "10", "19", "28"]

        features = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        return features

    def gram_matrix(self, input):
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
