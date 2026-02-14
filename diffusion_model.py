import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================================
# Residual Attention UNet Diffusion Refiner
# Inspired by:
# - AnoDDPM (Wolleb et al., MICCAI 2022)
# - Latent Diffusion Models (Rombach et al., CVPR 2022)
# ===============================================


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key   = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, C, H, W = x.shape

        q = self.query(x).view(B, -1, H * W)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),

            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class DiffusionUNet(nn.Module):
    """
     Best Diffusion Refinement UNet
    - Residual blocks
    - Self-attention bottleneck
    - Learns residual correction
    """

    def __init__(self, channels=3, base_dim=64):
        super().__init__()

        self.enc1 = nn.Conv2d(channels, base_dim, 3, padding=1)
        self.res1 = ResidualBlock(base_dim)

        self.enc2 = nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1)
        self.res2 = ResidualBlock(base_dim * 2)

        
        self.attn = SelfAttention(base_dim * 2)

        
        self.dec1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.res3 = ResidualBlock(base_dim)

        self.out = nn.Conv2d(base_dim, channels, 3, padding=1)

    def forward(self, x, t=None):

        x1 = F.silu(self.enc1(x))
        x1 = self.res1(x1)

        x2 = F.silu(self.enc2(x1))
        x2 = self.res2(x2)

       
        x2 = self.attn(x2)

        d1 = F.silu(self.dec1(x2))
        d1 = self.res3(d1)

        
        return self.out(d1)
