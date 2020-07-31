import pytorch_tools.models as models

ENCODER_SHAPES = {
    "vgg11_bn": (512, 512, 512, 256, 128),
    "vgg13_bn": (512, 512, 512, 256, 128),
    "vgg16_bn": (512, 512, 512, 256, 128),
    "vgg19_bn": (512, 512, 512, 256, 128),
    "resnet18": (512, 256, 128, 64, 64),
    "resnet34": (512, 256, 128, 64, 64),
    "resnet50": (2048, 1024, 512, 256, 64),
    "resnet101": (2048, 1024, 512, 256, 64),
    "resnet152": (2048, 1024, 512, 256, 64),
    "resnext50_32x4d": (2048, 1024, 512, 256, 64),
    "se_resnet34": (512, 256, 128, 64, 64),
    "se_resnet50": (2048, 1024, 512, 256, 64),
    "se_resnet101": (2048, 1024, 512, 256, 64),
    "se_resnet152": (2048, 1024, 512, 256, 64),
    "se_resnext50_32x4d": (2048, 1024, 512, 256, 64),
    "se_resnext101_32x4d": (2048, 1024, 512, 256, 64),
    "densenet121": (1024, 1024, 512, 256, 64),
    "densenet169": (1664, 1280, 512, 256, 64),
    "densenet201": (1920, 1792, 512, 256, 64),
    #'densenet161': (2208, 2112, 768, 384, 96),
    "efficientnet_b0": (320, 112, 40, 24, 16),
    "efficientnet_b1": (320, 112, 40, 24, 16),
    "efficientnet_b2": (352, 120, 48, 24, 16),
    "efficientnet_b3": (384, 136, 48, 32, 24),
    "efficientnet_b4": (448, 160, 56, 32, 24),
    "efficientnet_b5": (512, 176, 64, 40, 24),
    "efficientnet_b6": (576, 200, 72, 40, 32),
    "efficientnet_b7": (640, 224, 80, 48, 32),
    # this models return feature maps at OS= 32, 16, 8, 4, 4
    # they CAN'T be used as encoders in Unet and Linknet
    "tresnetm": (2048, 1024, 128, 64, 64),
    "tresnetl": (2432, 1216, 152, 76, 76),
    "tresnetxl": (2656, 1328, 166, 83, 83),
    # this models return feature maps at OS= 32, 16, 8, 4, 4
    # they CAN'T be used as encoders in Unet and Linknet
    "hrnet_w18_small": (144, 72, 36, 18, 18),
    "hrnet_w18": (144, 72, 36, 18, 18),
    "hrnet_w30": (240, 120, 60, 30, 30),
    "hrnet_w32": (256, 128, 64, 32, 32),
    "hrnet_w40": (320, 160, 80, 40, 40),
    "hrnet_w44": (352, 176, 88, 44, 44),
    "hrnet_w48": (384, 192, 96, 48, 48),
    "hrnet_w64": (512, 256, 128, 64, 64),
    # my custom model
    "bresnet50": (2048, 1024, 512, 256, 64),
}


def get_encoder(name, **kwargs):
    if name not in models.__dict__:
        raise ValueError(f"No such encoder: {name}")
    kwargs["encoder"] = True
    kwargs["pretrained"] = kwargs.pop("encoder_weights")
    m = models.__dict__[name](**kwargs)
    m.out_shapes = ENCODER_SHAPES[name]
    return m
