import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        ch = y.size(1)
        sigma, mu = torch.split(y.unsqueeze(-1).unsqueeze(-1), [ch // 2, ch // 2], dim=1)

        x_mu = x.mean(dim=[2, 3], keepdim=True)
        x_sigma = x.std(dim=[2, 3], keepdim=True)

        return sigma * ((x - x_mu) / x_sigma) + mu
