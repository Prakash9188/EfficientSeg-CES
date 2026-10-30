"""Small from-scratch fully convolutional network for binary foreground segmentation."""
import torch
from torch import nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparable(nn.Sequential):
    """Depthwise spatial convolution followed by a pointwise channel mixer."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class TinySegNet(nn.Module):
    """
    Lightweight encoder-decoder. Input/output are N x 3 x H x W and N x 2 x H x W.
    It is intentionally trained from scratch and uses no pretrained components.
    """
    def __init__(self, width=16):
        super().__init__()
        w = width
        self.stem = ConvBNReLU(3, w, 3, 1)
        self.enc1 = nn.Sequential(DepthwiseSeparable(w, 2*w, 2),
                                  DepthwiseSeparable(2*w, 2*w, 1))
        self.enc2 = nn.Sequential(DepthwiseSeparable(2*w, 3*w, 2),
                                  DepthwiseSeparable(3*w, 3*w, 1))
        self.enc3 = nn.Sequential(DepthwiseSeparable(3*w, 4*w, 2),
                                  DepthwiseSeparable(4*w, 4*w, 1))
        self.bottleneck = DepthwiseSeparable(4*w, 4*w, 1)

        self.dec2 = nn.Sequential(ConvBNReLU(4*w + 3*w, 3*w, 1),
                                  DepthwiseSeparable(3*w, 3*w))
        self.dec1 = nn.Sequential(ConvBNReLU(3*w + 2*w, 2*w, 1),
                                  DepthwiseSeparable(2*w, 2*w))
        self.dec0 = nn.Sequential(ConvBNReLU(2*w + w, w, 1),
                                  DepthwiseSeparable(w, w))
        self.head = nn.Conv2d(w, 2, kernel_size=1)

    @staticmethod
    def _up(x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        z = self.bottleneck(s3)
        z = self.dec2(torch.cat([self._up(z, s2), s2], dim=1))
        z = self.dec1(torch.cat([self._up(z, s1), s1], dim=1))
        z = self.dec0(torch.cat([self._up(z, s0), s0], dim=1))
        return self.head(z)
