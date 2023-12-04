from denoising_diffusion_pytorch import Unet
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import src.util.util as ut

# Neural network models. For complex images, use Unet.

class UNet(Unet):
    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)

    def forward(self, t, x):
        # call the unet with x and t inverted in order as input
        if t.dim() == 0:
            batch_size = x.shape[0]
            t = t.expand(batch_size)
        return super().forward(x, t)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SmallNet(nn.Module):
    def __init__(self, dim, num_channels=1, sinusoidal_pos_emb_theta = 10000):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1),
        )

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, num_channels*2)
        )

        self.act = nn.Sigmoid()
    
    def forward(self, t, x):
        # t: (b)
        # x: (b, c, h, w)
        # output: (b, c, h, w)
       
        time_emb = self.time_mlp(t)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim = 1)

        x = self.cnn(x) * (scale + 1) + shift
        x = self.act(x)
        return x

class LatentClassifier(nn.Module):
    '''
    A classifier for image data building a low-dimensional latent representation.
    '''
    def __init__(self, z_dim, ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
                        nn.Conv2d(1, 28, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(28),
                        nn.Conv2d(28, 28, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),

                        nn.Conv2d(28, 56, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(56),
                        nn.Conv2d(56, 56, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),

                        nn.Conv2d(56, 112, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(112),
                        nn.Conv2d(112, 112, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),

                        nn.Flatten(),
                        nn.Linear(112 * 3 * 3, z_dim*2),
                        nn.ReLU(),
                    )

        self.z_to_logits = nn.Sequential(
                        nn.Linear(z_dim, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10),
                    )
        
    def forward(self, x):
        h = self.encoder(x)
        z_mu, z_sigma = ut.gaussian_parameters(h)
        z = z_mu + z_sigma * torch.randn_like(z_sigma)

        logits = self.z_to_logits(z)
        return logits, z_mu, z_sigma


