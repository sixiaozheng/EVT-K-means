import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy.stats import genextreme, genpareto

def gev_pdf(x, c, loc, scale):

    x_ = (x - loc) / scale
    if c == 0.0:
        y = (1/scale)*troch.exp(-torch.exp(-x_)) * torch.exp(-x_)
    else:
        y = (1/scale)*torch.exp(-torch.pow(1 - c * x_, 1 / c)) * torch.pow(1 - c * x_, 1 / c - 1)
    y[torch.isnan(y)] = 0.0
    y[y<0.0] = 0.0
    return y

def gpd_pdf(x, c, loc, scale):

    x_ = (x - loc) / scale
    if c > 0:
        y = (1 / scale) * torch.pow(1 + c * x_, -1 - 1 / c)
        y[x_ < 0] = 0.0
    elif c < 0:
        y = (1 / scale) * torch.pow(1 + c * x_, -1 - 1 / c)
        y[(x_ < 0) | (x_ > -1 / c)] = 0.0
    elif c == 0:
        y = (1 / scale) * torch.exp(-x_)
        y[x_ < 0] = 0.0
    else:
        pass
    y[torch.isnan(y)] = 0.0
    y[y<0.0] = 0.0
    return y

class EVT(nn.Module):
    def __init__(self, n_clusters, extreme_model, device):
        super(EVT, self).__init__()
        
        self.n_clusters = n_clusters
        self.extreme_model = extreme_model
        self.device = device
        self.shape =  nn.Parameter(torch.ones(1,self.n_clusters,dtype=torch.float32,requires_grad=True))
        self.loc = nn.Parameter(torch.zeros(1,self.n_clusters,dtype=torch.float32,requires_grad=True))
        self.scale = nn.Parameter(torch.ones(1,self.n_clusters,dtype=torch.float32,requires_grad=True))

    def forward(self, x):
        if self.extreme_model == 'gev':
            y = []
            for idx, x_i in enumerate(x):
                y_i = self.gev_pdf(x_i, -self.shape[0,idx], self.loc[0,idx], self.scale[0,idx])
                y.append(y_i)
            return y

        elif self.extreme_model == 'gpd':
            y = []
            for idx, x_i in enumerate(x):
                y_i = self.gpd_pdf(x_i, self.shape[0,idx], self.loc[0,idx], self.scale[0,idx])
                y.append(y_i)
            return y

    def cdf(self, x):
        if self.extreme_model == 'gev':
            c = self.shape.detach().cpu().numpy()
            loc = self.loc.detach().cpu().numpy()
            scale = self.scale.detach().cpu().numpy()
            x_ = x.cpu().numpy()
            prob = np.zeros_like(x_)
            for i in range(self.n_clusters):
                prob[:,i] = genextreme.cdf(x_[:,i], c[0,i], loc[0,i], scale[0,i])
            return torch.from_numpy(prob).to(self.device)

        elif self.extreme_model == 'gpd':
            c = self.shape.detach().cpu().numpy()
            loc = self.loc.detach().cpu().numpy()
            scale = self.scale.detach().cpu().numpy()
            x_ = x.cpu().numpy()
            prob = np.zeros_like(x_)
            for i in range(self.n_clusters):
                prob[:,i] = genpareto.cdf(x_[:,i], c[0,i], loc[0,i], scale[0,i])
            return torch.from_numpy(prob).to(self.device)

    def gev_pdf(self, x, c, loc, scale):
        x_ = (x - loc) / scale
        if c == 0.0:
            y = (1/scale)*troch.exp(-torch.exp(-x_)) * torch.exp(-x_)
        else:
            y = (1/scale)*torch.exp(-torch.pow(1 - c * x_, 1 / c)) * torch.pow(1 - c * x_, 1 / c - 1)
        y[torch.isnan(y)] = 0.0
        y[y<0.0] = 0.0
        return y

    def gpd_pdf(self,x, c, loc, scale):
        x_ = (x - loc) / scale
        if c > 0.0:
            y = (1 / scale) * torch.pow(1 + c * x_, -1 - 1 / c)
            y[x_ < 0] = 0.0
        elif c < 0.0:
            y = (1 / scale) * torch.pow(1 + c * x_, -1 - 1 / c)
            y[(x_ < 0) | (x_ > -1 / c)] = 0.0
        elif c == 0.0:
            y = (1 / scale) * torch.exp(-x_)
            y[x_ < 0] = 0.0
        else:
            pass
        y[torch.isnan(y)] = 0.0
        y[y<0.0] = 0.0
        return y