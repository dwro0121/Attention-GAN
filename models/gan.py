import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .spectral import SpectralNorm
from .attention_layer import Attention_Layer


class G(nn.Module):
    def __init__(self, z_dim=100, attention="SA"):
        super(G, self).__init__()
        self.layer1 = nn.Sequential(
            *[
                SpectralNorm(nn.ConvTranspose2d(z_dim, 512, 4)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ]
        )
        self.layer2 = nn.Sequential(
            *[
                SpectralNorm(nn.ConvTranspose2d(512, 256, 4, 2, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ]
        )
        self.layer3 = nn.Sequential(
            *[
                SpectralNorm(nn.ConvTranspose2d(256, 128, 4, 2, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )
        self.layer4 = nn.Sequential(
            *[
                SpectralNorm(nn.ConvTranspose2d(128, 64, 4, 2, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ]
        )
        self.out = nn.Sequential(
            *[
                nn.ConvTranspose2d(64, 3, 4, 2, 1),
                nn.Tanh(),
            ]
        )

        self.attention = Attention_Layer(64, attention)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.attention(out)
        out = self.out(out)

        return out


class D(nn.Module):
    def __init__(self, attention="SA"):
        super(D, self).__init__()
        self.layer1 = nn.Sequential(
            *[
                SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1)),
                nn.LeakyReLU(0.1),
            ]
        )
        self.layer2 = nn.Sequential(
            *[
                SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.LeakyReLU(0.1),
            ]
        )
        self.layer3 = nn.Sequential(
            *[
                SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.LeakyReLU(0.1),
            ]
        )
        self.layer4 = nn.Sequential(
            *[
                SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)),
                nn.LeakyReLU(0.1),
            ]
        )
        self.last = nn.Conv2d(512, 1, 4)

        self.attention = Attention_Layer(64, attention)

    def forward(self, x):
        out = self.layer1(x)
        out = self.attention(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out.squeeze()
