import torch
import torch.nn as nn

from pytorch_tools.modules import activation_from_name


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, activation="identity"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.activation = activation_from_name(activation)

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            x = self.activation(x)
        return x
