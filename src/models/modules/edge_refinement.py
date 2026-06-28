import torch
import torch.nn as nn

class EdgeRefinementModule(nn.Module):
    """Lightweight edge refinement block for semantic segmentation.

    Can be attached after a decoder to sharpen boundaries with minimal
    computational overhead.
    """
    def __init__(self, channels:int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        residual = x
        out = self.block(x)
        return self.activation(out + residual)
