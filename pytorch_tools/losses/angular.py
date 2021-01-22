import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss


class AdaCos(Loss):
    """PyTorch implementation of AdaCos. See Ref[1] for paper

    This implementation is different from the most open-source implementations in following ways:
    1) expects raw logits of size (bs x num_classes) not (bs, embedding_size)
    2) despite AdaCos being dynamic, still add an optional margin parameter
    3) calculate running average stats of B and θ, not batch-wise stats as in original paper
    4) normalize input logits, not embeddings and weights

    Args:
        margin (float): margin in radians
        momentum (float): momentum for running average of B and θ

    Input:
        y_pred (torch.Tensor): shape BS x N_classes
        y_true (torch.Tensor): one-hot encoded. shape BS x N_classes
    Reference:
        [1] Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations

    """

    def __init__(self, embedding_size, num_classes, final_criterion, margin=0, momentum=0.95):
        super(AdaCos, self).__init__()
        self.final_criterion = final_criterion
        self.margin = margin
        self.momentum = momentum
        self.prev_s = 10
        self.running_B = 1000  # default value is chosen so that initial S is ~10
        self.running_theta = math.pi / 4
        self.eps = 1e-7
        self.register_parameter("weight", torch.nn.Parameter(torch.zeros(num_classes, embedding_size)))
        nn.init.xavier_uniform_(self.weight)

        self.idx = 0

    def forward(self, embedding, y_true):

        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + self.eps, 1 - self.eps)
        # cos_theta = torch.cos(torch.acos(cos_theta + self.margin))

        if y_true.dim() != 1:
            y_true_one_hot = y_true.float()
        else:
            y_true_one_hot = torch.zeros_like(cos_theta)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)

        with torch.no_grad():
            B_batch = cos_theta[y_true_one_hot.eq(0)].mul(self.prev_s).exp().sum().div(embedding.size(0))
            self.running_B = self.running_B * self.momentum + B_batch * (1 - self.momentum)
            theta = torch.acos(cos_theta.clamp(-1 + self.eps, 1 - self.eps))
            # originally authors use median, but I use mean
            theta_batch = theta[y_true_one_hot.ne(0)].mean().clamp_max(math.pi / 4)
            self.running_theta = self.running_theta * self.momentum + theta_batch * (1 - self.momentum)
            self.prev_s = self.running_B.log() / torch.cos(self.running_theta)

        self.idx += 1
        if self.idx % 1000 == 0:
            print(
                f"\nRunning B: {self.running_B:.2f}. Running theta: {self.running_theta:.2f}. Running S: {self.prev_s:.2f}"
            )

        return self.final_criterion(cos_theta * self.prev_s, y_true_one_hot)
