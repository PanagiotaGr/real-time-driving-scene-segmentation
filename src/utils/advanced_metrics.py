from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    support: int


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute a semantic-segmentation confusion matrix.

    Rows correspond to ground-truth classes and columns correspond to predicted
    classes. Input tensors are expected to contain integer class indices with
    shape (N, H, W) or any flattenable shape.
    """
    preds = preds.detach().view(-1).long().cpu()
    targets = targets.detach().view(-1).long().cpu()

    if ignore_index is not None:
        valid = targets != ignore_index
        preds = preds[valid]
        targets = targets[valid]

    valid = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    encoded = targets * num_classes + preds
    matrix = torch.bincount(encoded, minlength=num_classes ** 2)
    return matrix.reshape(num_classes, num_classes)


def per_class_metrics(matrix: torch.Tensor, eps: float = 1e-7) -> Dict[int, PerClassMetrics]:
    """Return precision, recall, F1, IoU and support for each class."""
    matrix = matrix.float()
    true_positive = torch.diag(matrix)
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive
    support = matrix.sum(dim=1)

    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = true_positive / (true_positive + false_positive + false_negative + eps)

    output: Dict[int, PerClassMetrics] = {}
    for class_id in range(matrix.shape[0]):
        output[class_id] = PerClassMetrics(
            precision=float(precision[class_id].item()),
            recall=float(recall[class_id].item()),
            f1=float(f1[class_id].item()),
            iou=float(iou[class_id].item()),
            support=int(support[class_id].item()),
        )
    return output


def macro_metrics(metrics: Dict[int, PerClassMetrics]) -> Dict[str, float]:
    """Compute macro-averaged precision, recall, F1 and IoU."""
    values = list(metrics.values())
    if not values:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "macro_iou": 0.0}

    return {
        "macro_precision": float(np.mean([m.precision for m in values])),
        "macro_recall": float(np.mean([m.recall for m in values])),
        "macro_f1": float(np.mean([m.f1 for m in values])),
        "macro_iou": float(np.mean([m.iou for m in values])),
    }


def metrics_to_rows(
    metrics: Dict[int, PerClassMetrics],
    class_names: Optional[List[str]] = None,
) -> List[dict]:
    """Convert class metrics to CSV/JSON-friendly rows."""
    rows: List[dict] = []
    for class_id, metric in metrics.items():
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": metric.precision,
                "recall": metric.recall,
                "f1": metric.f1,
                "iou": metric.iou,
                "support": metric.support,
            }
        )
    return rows
