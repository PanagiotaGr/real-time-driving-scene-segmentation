# Experiment Log

This file is the running laboratory notebook for the project. Every training run, failed run, evaluation, and benchmark should be recorded here.

## How to Use This Log

Create a new section for every experiment using the format below. Use short experiment IDs such as `EXP001`, `EXP002`, and so on.

---

## EXP001 - Baseline U-Net on CamVid

### Status

Planned

### Research Question

How well does a standard U-Net baseline perform on the selected driving-scene segmentation dataset?

### Hypothesis

U-Net should provide a strong accuracy baseline but may be slower than lightweight real-time models.

### Configuration

| Field | Value |
|---|---|
| Date |  |
| Git commit |  |
| Dataset | CamVid |
| Split | train/val |
| Model | U-Net |
| Input resolution |  |
| Number of classes |  |
| Loss | Cross Entropy |
| Optimizer |  |
| Learning rate |  |
| Batch size |  |
| Epochs |  |
| Random seed |  |
| Device |  |

### Commands

```bash
python src/training/train.py
```

```bash
python src/training/evaluate.py results/checkpoints/best.pth
```

### Results

| Metric | Value |
|---|---:|
| mIoU |  |
| Pixel Accuracy |  |
| Mean latency (ms) |  |
| FPS |  |
| Parameters |  |
| Peak GPU memory |  |

### Observations

- 

### Failure Cases

- 

### Next Step

- 

---

## EXP002 - ENet-like Real-Time Baseline

### Status

Planned

### Research Question

How much accuracy is lost when using a lightweight model designed for faster inference?

### Hypothesis

The ENet-like model should improve FPS and latency, but may reduce class-wise IoU for small or thin objects.

### Results Summary

| Metric | U-Net | ENet-like | Difference |
|---|---:|---:|---:|
| mIoU |  |  |  |
| Pixel Accuracy |  |  |  |
| FPS |  |  |  |
| Latency |  |  |  |
| Parameters |  |  |  |

### Interpretation

- 

---

## EXP003 - Hybrid Loss Ablation

### Status

Planned

### Research Question

Does a hybrid segmentation loss improve class imbalance and boundary quality compared with Cross Entropy alone?

### Hypothesis

A combined CE + Dice or CE + Dice + Boundary loss should improve minority-class IoU, especially for small structures.

### Results Summary

| Loss | mIoU | Small-object IoU | Boundary Quality Notes | FPS Impact |
|---|---:|---:|---|---:|
| Cross Entropy |  |  |  |  |
| CE + Dice |  |  |  |  |
| CE + Dice + Boundary |  |  |  |  |

---

## EXP004 - Robustness Stress Test

### Status

Planned

### Research Question

How does each model degrade under visual corruptions such as blur, darkness, fog, noise, and reduced contrast?

### Hypothesis

Lightweight models may degrade more strongly under domain shift because they have lower representational capacity.

### Results Summary

| Model | Clean | Blur | Noise | Darkness | Fog | Mean Drop |
|---|---:|---:|---:|---:|---:|---:|
| U-Net |  |  |  |  |  |  |
| ENet-like |  |  |  |  |  |  |
| Proposed |  |  |  |  |  |  |

---

## General Notes

Use this section for observations that affect multiple experiments.

- 
