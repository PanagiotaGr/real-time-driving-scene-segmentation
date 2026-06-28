import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.losses.boundary_loss import BoundaryLoss


class DiceLoss(nn.Module):
    """Multi-class Dice loss for semantic segmentation."""

    def __init__(self, smooth: float = 1.0, ignore_index: int | None = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid_mask = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target.clone()
            target[~valid_mask] = 0

        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class HybridSegmentationLoss(nn.Module):
    """Weighted Cross-Entropy + Dice + Boundary loss.

    This loss supports ablation experiments through configurable weights:

    - alpha: Cross-Entropy contribution.
    - beta: Dice contribution.
    - gamma: Boundary contribution.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.boundary = BoundaryLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, target.long())
        dice_loss = self.dice(logits, target)
        boundary_loss = self.boundary(logits, target)
        return self.alpha * ce_loss + self.beta * dice_loss + self.gamma * boundary_loss

    def components(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return individual loss components for experiment logging."""
        ce_loss = self.cross_entropy(logits, target.long())
        dice_loss = self.dice(logits, target)
        boundary_loss = self.boundary(logits, target)
        total = self.alpha * ce_loss + self.beta * dice_loss + self.gamma * boundary_loss
        return {
            "total": total,
            "cross_entropy": ce_loss,
            "dice": dice_loss,
            "boundary": boundary_loss,
        }
