import torch

from src.utils.losses import CombinedLoss, DiceLoss
from src.utils.losses.hybrid_loss import HybridSegmentationLoss


def test_dice_loss_returns_scalar():
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 3, (2, 16, 16))

    loss = DiceLoss()(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_combined_loss_backward():
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 3, (2, 16, 16))

    loss = CombinedLoss()(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(loss)


def test_hybrid_loss_returns_components():
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 3, (2, 16, 16))
    criterion = HybridSegmentationLoss(alpha=1.0, beta=0.5, gamma=0.1)

    components = criterion.components(logits, targets)

    assert set(components.keys()) == {"total", "cross_entropy", "dice", "boundary"}
    assert torch.isfinite(components["total"])
