"""Simple GAN architecture using PyTorch."""

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100, img_shape: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_shape),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_shape: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
