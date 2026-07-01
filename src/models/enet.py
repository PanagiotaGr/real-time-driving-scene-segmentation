import torch
import torch.nn as nn
import torch.nn.functional as F


class ENetBlock(nn.Module):
    """Small convolutional block used by the lightweight ENet-style baseline."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ENet(nn.Module):
    """Lightweight ENet-style semantic segmentation baseline.

    This implementation is intentionally compact and stable for experiments. It
    preserves the main real-time baseline idea: reduce computation in an encoder,
    then recover dense predictions with a lightweight decoder. The output spatial
    resolution always matches the input resolution.
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.enc1 = ENetBlock(n_channels, 32, stride=2)
        self.enc2 = ENetBlock(32, 64, stride=2)
        self.enc3 = ENetBlock(64, 128, stride=2)

        self.bottleneck = ENetBlock(128, 128)

        self.dec2 = ENetBlock(128 + 64, 64)
        self.dec1 = ENetBlock(64 + 32, 32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.classifier(d1)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
