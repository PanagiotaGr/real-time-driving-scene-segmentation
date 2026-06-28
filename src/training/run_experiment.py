import argparse
import csv
import json
import os
import time
from datetime import datetime

import torch

from src.config import get_config
from src.datasets import get_camvid_dataloaders
from src.models.bisenet import BiSeNet
from src.models.enet import ENet
from src.models.unet import UNet
from src.utils.benchmark import benchmark_inference, count_parameters
from src.utils.metrics import SegmentationMetrics


def build_model(config):
    model_name = config.model.model_name.lower()
    if model_name == "unet":
        return UNet(n_channels=config.model.input_channels, n_classes=config.model.num_classes)
    if model_name == "enet":
        return ENet(n_channels=config.model.input_channels, n_classes=config.model.num_classes)
    if model_name == "bisenet":
        return BiSeNet(n_channels=config.model.input_channels, n_classes=config.model.num_classes)
    raise ValueError(f"Unsupported model: {config.model.model_name}")


def safe_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def run_experiment(checkpoint_path):
    config = get_config()
    device = safe_device(config.training.device)

    model = build_model(config).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    _, val_loader = get_camvid_dataloaders(
        config.dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    metrics = SegmentationMetrics(num_classes=config.dataset.num_classes)
    all_preds = []
    all_targets = []
    total_forward_seconds = 0.0
    total_images = 0

    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        outputs = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_forward_seconds += time.perf_counter() - start
        total_images += images.shape[0]

        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    iou = metrics.iou_metrics(all_preds, all_targets)
    pixel_accuracy = metrics.pixel_accuracy(all_preds, all_targets)
    parameter_stats = count_parameters(model)

    height, width = config.dataset.image_size
    speed_stats = benchmark_inference(
        model=model,
        input_shape=(config.training.batch_size, config.model.input_channels, height, width),
        device=device,
        warmup_iters=10,
        benchmark_iters=50,
    )

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "experiment_name": config.experiment.experiment_name,
        "model_name": config.model.model_name,
        "dataset": "CamVid",
        "input_height": height,
        "input_width": width,
        "batch_size": config.training.batch_size,
        "device": str(device),
        "checkpoint_path": checkpoint_path,
        "mean_iou": float(iou["mIoU"]),
        "pixel_accuracy": float(pixel_accuracy),
        "class_iou": {str(k): float(v) for k, v in iou["class_iou"].items()},
        "eval_latency_ms_per_image": (total_forward_seconds / max(total_images, 1)) * 1000.0,
        "eval_fps": total_images / total_forward_seconds if total_forward_seconds > 0 else 0.0,
        **parameter_stats,
        **speed_stats,
    }

    output_dir = os.path.join(config.experiment.output_dir, config.experiment.experiment_name)
    save_json(os.path.join(output_dir, "experiment_results.json"), results)
    append_csv(
        os.path.join(config.experiment.output_dir, "all_experiments.csv"),
        {
            "timestamp": results["timestamp"],
            "experiment_name": results["experiment_name"],
            "model_name": results["model_name"],
            "dataset": results["dataset"],
            "resolution": f"{height}x{width}",
            "mean_iou": results["mean_iou"],
            "pixel_accuracy": results["pixel_accuracy"],
            "latency_ms_per_image": results["latency_ms_per_image"],
            "fps": results["fps"],
            "total_parameters": results["total_parameters"],
            "peak_memory_mb": results["peak_memory_mb"],
            "checkpoint_path": results["checkpoint_path"],
        },
    )

    print(json.dumps(results, indent=2))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run a reproducible segmentation experiment.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained model checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.checkpoint)
