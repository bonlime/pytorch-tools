import torch
import torch.nn as nn
import torch.nn.functional as F


def silu(input, beta=1):
    '''
    Applies the Sigmoid Linear Unit (SiLU) / Swish-1 function element-wise:
        SiLU(x) = x * sigmoid(beta * x)
    '''
    # return input.mul_(torch.sigmoid(input))
    if not isinstance(beta, torch.Tensor):
        beta = torch.Tensor([beta])
    
    return input * torch.sigmoid(input.mul_(beta))



class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(beta * x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
        https://arxiv.org/pdf/1710.05941.pdf
    Examples:
        >>> m = SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self, beta=1):
        '''
        Init method.
        '''
        super().__init__()  # init the base class
        self.beta = beta

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return silu(input, self.beta)  # simply apply already implemented SiLU

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class SoftExponential(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(SoftExponential, self).__init__()
        self.in_features = in_features

        # initialize alpha
        if alpha is None:
            self.alpha = Parameter(torch.tensor(0.0))  # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.tensor(alpha))  # create a tensor out of alpha
            
        self.alpha.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha