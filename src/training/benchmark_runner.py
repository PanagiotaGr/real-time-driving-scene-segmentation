import argparse
import csv
import json
import os
from typing import Dict, List

from src.training.run_experiment import run_experiment


def read_manifest(path: str) -> List[Dict[str, str]]:
    """Read benchmark manifest from JSON.

    Expected format:
    [
      {"name": "unet_ce", "checkpoint": "results/checkpoints/unet_ce.pth"},
      {"name": "unet_hybrid", "checkpoint": "results/checkpoints/unet_hybrid.pth"}
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list):
        raise ValueError("Benchmark manifest must be a JSON list.")

    for item in manifest:
        if "name" not in item or "checkpoint" not in item:
            raise ValueError("Each manifest item must contain 'name' and 'checkpoint'.")

    return manifest


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_result(experiment_name: str, result: Dict[str, object]) -> Dict[str, object]:
    return {
        "experiment_name": experiment_name,
        "model_name": result.get("model_name"),
        "dataset": result.get("dataset"),
        "resolution": f"{result.get('input_height')}x{result.get('input_width')}",
        "mean_iou": result.get("mean_iou"),
        "pixel_accuracy": result.get("pixel_accuracy"),
        "eval_latency_ms_per_image": result.get("eval_latency_ms_per_image"),
        "eval_fps": result.get("eval_fps"),
        "benchmark_latency_ms_per_image": result.get("latency_ms_per_image"),
        "benchmark_fps": result.get("fps"),
        "total_parameters": result.get("total_parameters"),
        "trainable_parameters": result.get("trainable_parameters"),
        "peak_memory_mb": result.get("peak_memory_mb"),
        "checkpoint_path": result.get("checkpoint_path"),
    }


def run_benchmark_manifest(manifest_path: str, output_csv: str) -> List[Dict[str, object]]:
    manifest = read_manifest(manifest_path)
    rows: List[Dict[str, object]] = []

    for item in manifest:
        name = item["name"]
        checkpoint = item["checkpoint"]
        if not os.path.exists(checkpoint):
            print(f"Skipping {name}: checkpoint not found at {checkpoint}")
            continue

        print(f"Running benchmark: {name}")
        result = run_experiment(checkpoint)
        rows.append(summarize_result(name, result))

    write_summary_csv(output_csv, rows)
    print(f"Saved benchmark summary to {output_csv}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple segmentation benchmarks from a manifest.")
    parser.add_argument("--manifest", required=True, help="Path to JSON benchmark manifest.")
    parser.add_argument(
        "--output-csv",
        default="results/experiments/benchmark_summary.csv",
        help="Path where the benchmark summary CSV will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark_manifest(args.manifest, args.output_csv)
