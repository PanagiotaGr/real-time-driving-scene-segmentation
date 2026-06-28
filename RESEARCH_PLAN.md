# Research Plan: Real-Time Driving Scene Semantic Segmentation

This document defines the research direction, experimental protocol, and paper-oriented roadmap for the project.

## 1. Research Motivation

Real-time semantic segmentation is a core perception task for autonomous driving. The main challenge is to maintain high segmentation accuracy while keeping inference latency low enough for deployment on constrained hardware. This repository will be developed as a reproducible research framework for studying the accuracy-speed trade-off in driving-scene segmentation.

## 2. Main Research Question

Can lightweight semantic segmentation architectures achieve competitive mIoU on driving-scene datasets while preserving real-time inference performance?

Secondary questions:

- Which architecture provides the best trade-off between mIoU, latency, parameters, and GPU memory usage?
- Which object classes are most affected by lightweight model compression?
- Can boundary-aware or class-imbalance-aware training improve small-object segmentation without reducing FPS significantly?
- How robust are the models under visual domain shifts such as night, rain, fog, or low contrast?

## 3. Baseline Models

Initial experiments should compare at least the following model families:

- U-Net as a classical encoder-decoder baseline.
- ENet as a lightweight real-time baseline.
- BiSeNet-style bilateral segmentation as a speed-accuracy baseline.

Future extensions can include:

- Fast-SCNN.
- DDRNet.
- PIDNet.
- SegFormer-B0.
- A proposed lightweight edge-aware model.

## 4. Datasets

Initial dataset:

- CamVid.

Recommended extensions:

- Cityscapes.
- BDD100K.
- Mapillary Vistas.

The dataset pipeline should preserve clear train, validation, and test splits. Any resizing, augmentation, normalization, or class remapping must be documented in the experiment configuration.

## 5. Evaluation Metrics

Each experiment should report both accuracy and efficiency metrics:

Accuracy metrics:

- Mean Intersection over Union (mIoU).
- Pixel accuracy.
- Class-wise IoU.
- Precision, recall, and F1-score where appropriate.
- Confusion matrix.

Efficiency metrics:

- FPS.
- Mean latency per image.
- Number of parameters.
- FLOPs or MACs.
- Peak GPU memory usage.
- Input resolution used during benchmarking.

## 6. Experimental Protocol

Each model should be trained using the same dataset split, input resolution, optimizer family, and evaluation script unless an ablation study explicitly changes one variable.

Minimum protocol:

1. Train each baseline model with the same dataset split.
2. Save the best checkpoint according to validation mIoU.
3. Evaluate the best checkpoint on the validation or test set.
4. Record accuracy, latency, FPS, memory usage, and parameter count.
5. Generate qualitative visualizations for representative easy, medium, and hard samples.
6. Store the experiment configuration and metrics in a reproducible results directory.

## 7. Proposed Research Contribution Ideas

The project can evolve from implementation to research contribution through one of the following directions.

### 7.1 Edge-Aware Refinement

Add a lightweight boundary refinement module after the decoder to improve thin structures such as poles, signs, pedestrians, and lane boundaries.

Hypothesis:

> Boundary-aware refinement improves class-wise IoU for small and thin objects with limited latency overhead.

### 7.2 Hybrid Loss Function

Combine cross-entropy, Dice loss, and boundary-aware loss to address class imbalance and small-object segmentation.

Hypothesis:

> A hybrid loss improves minority-class segmentation compared with plain cross-entropy.

### 7.3 Robustness Analysis

Evaluate models under synthetic adverse conditions such as blur, rain, fog, noise, reduced brightness, and contrast shifts.

Hypothesis:

> Lightweight models degrade differently under visual domain shift, and this degradation can be quantified per class.

### 7.4 Knowledge Distillation

Use a larger teacher model to train a smaller real-time student model.

Hypothesis:

> Distillation improves the mIoU of a compact model while preserving real-time inference speed.

## 8. Ablation Study Plan

Possible ablations:

- Baseline model vs. baseline plus edge refinement.
- Cross-entropy vs. hybrid loss.
- Different input resolutions.
- Different augmentation strategies.
- Different decoder widths.
- With and without knowledge distillation.

Each ablation should change only one major factor at a time.

## 9. Paper Outline

Working title:

> Real-Time Semantic Segmentation for Driving Scenes: A Reproducible Accuracy-Speed Benchmark and Lightweight Refinement Study

Suggested structure:

1. Abstract.
2. Introduction.
3. Related Work.
4. Methodology.
5. Experimental Setup.
6. Results and Discussion.
7. Ablation Studies.
8. Limitations.
9. Conclusion and Future Work.

## 10. Reproducibility Checklist

Before submission or thesis writing, the repository should include:

- Installation instructions.
- Dataset preparation instructions.
- Training commands.
- Evaluation commands.
- Inference commands.
- YAML configuration files for every reported experiment.
- Random seed documentation.
- Hardware information.
- Exact metrics table.
- Saved qualitative examples.
- Clear license.

## 11. Immediate Next Steps

1. Verify that all current training and evaluation commands run successfully.
2. Add a standard results table template.
3. Add FPS and latency benchmarking utilities.
4. Add confusion matrix and class-wise IoU export.
5. Run the first baseline experiment on CamVid.
6. Document all results in a reproducible experiment log.
