import pytorch_tools.models as models

encoders = {}
encoders.update(vgg_encoders)
encoders.update(resnet_encoders)
encoders.update(resnext_encoders)
encoders.update(senet_encoders)



def get_encoder(name, **kwargs):
    if name not in models.__dict__:
        raise ValueError('No such encoder: {}'.format(name))
    kwargs['encoder'] = True
    kwargs['pretrained'] = kwargs.pop('encoder_weights')
    m = models.__dict__[name](**kwargs)
    return m