import torch
import numpy as np

class SegmentationMetrics:
    """
    Metric class for semantic segmentation evaluation.
    """
    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def pixel_accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes Pixel Accuracy.
        Args:
            preds: (N, H, W) - Predicted class indices.
            targets: (N, H, W) - Ground truth class indices.
        """
        correct = (preds == targets).float().sum()
        total = targets.numel()

        # If we have an ignore index, we should exclude it from the total
        if self.ignore_index != -1:
            mask = (targets != self.ignore_index)
            correct = (preds[mask] == targets[mask]).float().sum()
            total = mask.sum().float()

        return (correct / (total + 1e-7)).item()

    def iou_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Computes IoU (Intersection over Union) per class and mean IoU.
        Args:
            preds: (N, H, W) - Predicted class indices.
            targets: (N, H, W) - Ground truth class indices.
        """
        # Flatten for easier calculation
        preds = preds.view(-1)
        targets = targets.view(-1)

        ious = []
        class_ious = {}

        for cls in range(self.num_classes):
            # Mask for current class in ground truth
            gt_mask = (targets == cls)
            # Mask for current class in prediction
            pred_mask = (preds == cls)

            intersection = (gt_mask & pred_mask).float().sum()
            union = (gt_mask | pred_mask).float().sum()

            if union == 0:
                iou = float('nan') # Avoid division by zero
            else:
                iou = (intersection / union).item()

            if not np.isnan(iou):
                ious.append(iou)
                class_ious[cls] = iou
            else:
                class_ious[cls] = 0.0 # Or some other indicator

        mean_iou = np.mean(ious) if ious else 0.0

        return {
            "mIoU": mean_iou,
            "class_iou": class_ious
        }
