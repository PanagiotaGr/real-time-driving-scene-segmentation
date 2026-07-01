import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj)!r}")
    return asdict(obj)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def collect_environment() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": get_git_commit(),
    }


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_history_csv(history: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    num_rows = max((len(values) for values in history.values()), default=0)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for idx in range(num_rows):
            row = [idx + 1]
            for key in keys:
                values = history.get(key, [])
                row.append(values[idx] if idx < len(values) else "")
            writer.writerow(row)


def final_metrics_from_history(history: dict[str, list[float]]) -> dict[str, Any]:
    metrics = {}
    for key, values in history.items():
        if values:
            metrics[key] = values[-1]
    if history.get("val_mIoU"):
        metrics["best_val_mIoU"] = max(history["val_mIoU"])
    if history.get("val_loss"):
        metrics["best_val_loss"] = min(history["val_loss"])
    return metrics


def export_experiment_artifacts(config: Any, history: dict[str, list[float]]) -> Path:
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = dataclass_to_dict(config)
    metrics = final_metrics_from_history(history)
    environment = collect_environment()

    save_yaml(config_dict, output_dir / "config.yaml")
    save_yaml(history, output_dir / "history.yaml")
    save_json(metrics, output_dir / "metrics.json")
    save_json(environment, output_dir / "environment.json")
    save_history_csv(history, output_dir / "metrics.csv")

    return output_dir
