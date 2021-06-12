import torch.nn as nn
import torch.nn.functional as F
import torch


class EVTLoss(nn.Module):

    def __init__(self):
        super(EVTLoss, self).__init__()

    def forward(self, x):
        x = torch.cat(x)
        x = x + 1e-6
        loss = -x.log().sum()
        # loss = torch.mean((x - y)**2)
        return loss
