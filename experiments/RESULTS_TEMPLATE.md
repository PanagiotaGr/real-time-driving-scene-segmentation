# Experiment Results Template

Use this file to record every experiment in a consistent, paper-ready format.

## Experiment Metadata

| Field | Value |
|---|---|
| Experiment ID |  |
| Date |  |
| Research question |  |
| Dataset |  |
| Dataset split |  |
| Model |  |
| Input resolution |  |
| Number of classes |  |
| Random seed |  |
| Hardware |  |
| Training time |  |
| Checkpoint path |  |

## Training Configuration

| Component | Value |
|---|---|
| Optimizer |  |
| Initial learning rate |  |
| Scheduler |  |
| Batch size |  |
| Epochs |  |
| Loss function |  |
| Data augmentation |  |
| Image normalization |  |
| Early stopping |  |

## Accuracy Metrics

| Metric | Value |
|---|---:|
| Mean IoU |  |
| Pixel accuracy |  |
| Mean precision |  |
| Mean recall |  |
| Mean F1-score |  |

## Efficiency Metrics

| Metric | Value |
|---|---:|
| Parameters |  |
| FLOPs / MACs |  |
| Mean latency per image |  |
| FPS |  |
| Peak GPU memory |  |
| Benchmark batch size |  |
| Benchmark device |  |

## Class-wise IoU

| Class | IoU |
|---|---:|
| Road |  |
| Sidewalk |  |
| Building |  |
| Pole |  |
| Traffic light |  |
| Traffic sign |  |
| Vegetation |  |
| Sky |  |
| Person |  |
| Rider |  |
| Car |  |
| Truck |  |
| Bus |  |
| Motorcycle |  |
| Bicycle |  |

## Qualitative Results

Add representative examples for:

- Easy daytime scene.
- Dense traffic scene.
- Small-object scene.
- Low-light or adverse-condition scene.
- Failure case.

For each example, save:

1. Input image.
2. Ground-truth mask.
3. Predicted mask.
4. Error map.

## Observations

Describe the most important findings:

- Which classes improved?
- Which classes failed?
- Did the model satisfy the real-time constraint?
- What is the main accuracy-speed trade-off?
- What should be tested next?

## Paper-ready Summary

Write a concise paragraph that can later be reused in the Results section of the paper.

> 

