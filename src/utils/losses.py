import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C, H, W) - Predicted logits from the model.
            targets: (N, H, W) - Ground truth class indices.
        """
        num_classes = logits.shape[1]

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot encoding: (N, C, H, W)
        # We use one_hot and then permute to get (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Flatten tensors for calculation
        probs = probs.contiguous().view(-1)
        targets_one_hot = targets_one_hot.contiguous().view(-1)

        intersection = (probs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets_one_hot.sum() + self.smooth)

        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combines CrossEntropy and Dice Loss.
    """
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
