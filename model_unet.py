from __future__ import annotations
import torch
import torch.nn as nn

def z0(in_ch, out_ch, groups=8, dropout_p=0.0):
    z2 = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch), nn.SiLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch), nn.SiLU(inplace=True)]
    if dropout_p > 0:
        z2.append(nn.Dropout2d(dropout_p))
    return nn.Sequential(*z2)

class C1(nn.Module):

    def __init__(self, in_ch=10, out_ch=3, base=32):
        super().__init__()
        self.enc1 = z0(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = z0(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = z0(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.mid = z0(base * 4, base * 8, dropout_p=0.4)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = z0(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = z0(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = z0(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        z3 = self.enc1(x)
        z4 = self.enc2(self.pool1(z3))
        z5 = self.enc3(self.pool2(z4))
        z6 = self.mid(self.pool3(z5))
        z7 = self.up3(z6)
        z7 = torch.cat([z7, z5], dim=1)
        z7 = self.dec3(z7)
        z8 = self.up2(z7)
        z8 = torch.cat([z8, z4], dim=1)
        z8 = self.dec2(z8)
        z9 = self.up1(z8)
        z9 = torch.cat([z9, z3], dim=1)
        z9 = self.dec1(z9)
        return 0.1 * torch.tanh(self.out(z9))
UNetResidual=C1

