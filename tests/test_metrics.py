import torch

from src.utils.metrics import SegmentationMetrics


def test_pixel_accuracy_perfect_prediction():
    metrics = SegmentationMetrics(num_classes=3)
    preds = torch.tensor([[[0, 1], [2, 1]]])
    targets = torch.tensor([[[0, 1], [2, 1]]])

    assert metrics.pixel_accuracy(preds, targets) == 1.0


def test_iou_metrics_contains_mean_and_classes():
    metrics = SegmentationMetrics(num_classes=2)
    preds = torch.tensor([[[0, 1], [1, 1]]])
    targets = torch.tensor([[[0, 0], [1, 1]]])

    result = metrics.iou_metrics(preds, targets)

    assert "mIoU" in result
    assert "class_iou" in result
    assert set(result["class_iou"].keys()) == {0, 1}
    assert 0.0 <= result["mIoU"] <= 1.0
