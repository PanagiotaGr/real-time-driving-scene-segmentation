# Reproducibility Protocol

This document describes the minimum procedure required to reproduce the experiments in this repository in a scientifically traceable way.

## 1. Objective

The project studies real-time semantic segmentation for driving scenes. The central goal is to evaluate the trade-off between segmentation accuracy and inference efficiency.

The main quantities of interest are:

- Mean Intersection over Union (mIoU)
- Pixel Accuracy
- Class-wise IoU
- Inference latency
- Frames per second (FPS)
- Number of trainable parameters
- GPU memory usage, when available

## 2. Repository Version

Before running an experiment, record the exact Git commit:

```bash
git rev-parse HEAD
```

Every reported result should be linked to a commit hash. This prevents ambiguity when the code changes later.

## 3. Environment

Record the following information:

```bash
python --version
pip freeze > environment_freeze.txt
```

Also record:

- operating system
- CPU model
- GPU model, if used
- CUDA version, if used
- PyTorch version

## 4. Dataset Preparation

The expected CamVid-style structure is:

```text
data/camvid/
├── train/
│   ├── leftImg8bit/
│   └── gtSeg8bit/
└── val/
    ├── leftImg8bit/
    └── gtSeg8bit/
```

The image and mask files must be paired consistently. If the dataset is modified, resized, filtered, or remapped, the exact rule must be documented in the experiment log.

## 5. Training Reproduction

A baseline training run should include:

```bash
python src/training/train.py
```

For each run, record:

- model name
- random seed
- input resolution
- batch size
- optimizer
- learning rate
- loss function
- number of epochs
- checkpoint path

## 6. Evaluation Reproduction

After training, evaluate the saved checkpoint:

```bash
python src/training/evaluate.py results/checkpoints/best.pth
```

If the project path differs, use the actual checkpoint path created by the training script.

## 7. Speed Benchmarking

For fair FPS and latency reporting:

1. Use the same input resolution for all models.
2. Use the same hardware.
3. Run warm-up iterations before timing.
4. Report batch size.
5. Report mean latency, not only best-case latency.

Suggested reporting format:

| Model | Resolution | Device | Batch Size | Latency (ms) | FPS |
|---|---:|---|---:|---:|---:|
| U-Net |  |  |  |  |  |
| ENet |  |  |  |  |  |

## 8. Result Storage

Each experiment should produce a folder such as:

```text
results/experiments/EXP001_unet_baseline/
├── config.yaml
├── metrics.json
├── training_log.csv
├── environment_freeze.txt
├── qualitative_examples/
└── notes.md
```

## 9. Reproducibility Checklist

Before claiming that an experiment is reproduced, verify:

- [ ] The code commit is recorded.
- [ ] The dataset split is documented.
- [ ] The training configuration is saved.
- [ ] The random seed is recorded.
- [ ] The checkpoint path is saved.
- [ ] Quantitative metrics are exported.
- [ ] Qualitative examples are stored.
- [ ] Hardware and software versions are documented.
- [ ] Any deviation from the baseline protocol is explained.

## 10. Scientific Reporting Principle

Do not report a single number without context. For this project, an accuracy result is meaningful only when paired with the input resolution, dataset split, model variant, and hardware used for timing.
