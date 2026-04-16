import torch
import torch.nn as nn
import torch.nn.functional as F

class ENetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ENetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.stride = stride

    def forward(self, x):
        return self.conv(x)

class ENet(nn.Module):
    """
    ENet (Efficient Neural Network) implementation for semantic segmentation.
    Designed for efficient computation.
    """
    def __init__(self, n_channels=3, n_classes=32):
        super(ENet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Simplified ENet architecture for implementation
        # In a real scenario, this would follow the exact ENet paper specs
        self.enc1 = ENetBlock(n_channels, 32)
        self.enc2 = ENetBlock(32, 64, stride=2)
        self.enc3 = ENetBlock(64, 128, stride=2)

        self.bottleneck = ENetBlock(128, 256)

        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ENetBlock(128 + 128, 64) # Skip connection from enc2
        self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = ENetBlock(32 + 32, 32) # Skip connection from enc1

        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # Decoder with skip connections
        d1 = self.dec1(b)
        # Need to handle spatial dimensions if they don't match perfectly
        # but here we use simple stride-2 convtranspose
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1)

        d3 = self.dec3(d2)
        d4 = torch.cat([d3, e1], dim=1)
        d5 = self.dec4(d4)

        return self.classifier(d5)
