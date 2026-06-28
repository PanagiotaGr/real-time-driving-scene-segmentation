# Benchmark Table

This table is intended to become the central quantitative table for the thesis or paper.

## Main Accuracy-Speed Benchmark

| Model | Dataset | Resolution | mIoU (%) | Pixel Acc. (%) | Params (M) | FLOPs (G) | Latency (ms) | FPS | GPU Memory (MB) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| U-Net | CamVid |  |  |  |  |  |  |  |  | Baseline encoder-decoder |
| ENet | CamVid |  |  |  |  |  |  |  |  | Lightweight real-time baseline |
| BiSeNet | CamVid |  |  |  |  |  |  |  |  | Bilateral speed-accuracy baseline |
| Proposed | CamVid |  |  |  |  |  |  |  |  | Add module/loss name |

## Ablation Study Table

| Experiment | Model Variant | Loss | Edge Module | Augmentation | Resolution | mIoU (%) | FPS | Main Finding |
|---|---|---|---|---|---:|---:|---:|---|
| A0 | Baseline | Cross-Entropy | No | Standard |  |  |  | Reference model |
| A1 | Baseline + Hybrid Loss | CE + Dice | No | Standard |  |  |  | Tests class imbalance handling |
| A2 | Baseline + Edge Refinement | Cross-Entropy | Yes | Standard |  |  |  | Tests boundary improvement |
| A3 | Full Proposed | CE + Dice + Boundary | Yes | Standard |  |  |  | Tests full method |

## Robustness Benchmark

| Model | Clean mIoU | Rain mIoU | Fog mIoU | Night mIoU | Blur mIoU | Noise mIoU | Mean Drop | Robustness Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| U-Net |  |  |  |  |  |  |  |  |
| ENet |  |  |  |  |  |  |  |  |
| BiSeNet |  |  |  |  |  |  |  |  |
| Proposed |  |  |  |  |  |  |  |  |

## Reporting Rules

- Always report the input resolution used during benchmarking.
- Always specify the device used for FPS and latency.
- Run multiple warm-up iterations before measuring speed.
- Report batch size for both training and inference.
- Do not compare FPS numbers across different hardware as if they were directly equivalent.
- For paper tables, bold the best value and underline the second-best value only after all experiments are complete.

## Interpretation Guide

A model should not be judged only by mIoU. For real-time driving-scene segmentation, the key result is the trade-off between:

- segmentation accuracy,
- inference speed,
- memory footprint,
- robustness to domain shift,
- class-wise behavior on safety-critical objects.

Safety-critical classes such as pedestrians, riders, traffic signs, traffic lights, and vehicles should be discussed separately because global mIoU can hide poor performance on rare but important classes.
