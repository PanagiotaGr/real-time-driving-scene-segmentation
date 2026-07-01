import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """Lightweight boundary-aware loss for segmentation masks."""

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        pred = probs.max(dim=1, keepdim=True).values
        target = target.float().unsqueeze(1)

        px = F.conv2d(pred, self.sx, padding=1)
        py = F.conv2d(pred, self.sy, padding=1)
        tx = F.conv2d(target, self.sx, padding=1)
        ty = F.conv2d(target, self.sy, padding=1)

        pred_edge = torch.sqrt(px**2 + py**2 + 1e-6)
        target_edge = torch.sqrt(tx**2 + ty**2 + 1e-6)
        return F.l1_loss(pred_edge, target_edge)
