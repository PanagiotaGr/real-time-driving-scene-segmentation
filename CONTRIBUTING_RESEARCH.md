# Research Contribution Guidelines

This repository is structured as a research-oriented computer vision project. Contributions should improve reproducibility, experimental clarity, or model quality.

## 1. Contribution Types

Useful contributions include:

- new model architectures
- improved training or evaluation scripts
- reproducible experiment configurations
- additional metrics
- qualitative visualizations
- robustness analysis
- documentation improvements
- bug fixes that affect reproducibility

## 2. Experiment-First Workflow

Before adding a new method, define the research question.

Example:

> Does edge-aware refinement improve small-object and boundary segmentation without significantly reducing FPS?

Then define:

- baseline model
- proposed change
- controlled variables
- expected metric improvement
- possible failure cases

## 3. Branch Naming

Recommended branch names:

```text
experiment/unet-baseline
experiment/enet-baseline
feature/edge-refinement
feature/hybrid-loss
analysis/robustness-study
docs/reproduction-protocol
```

## 4. Commit Style

Use descriptive commits:

```text
Add ENet baseline model
Add class-wise IoU export
Document CamVid preprocessing
Fix mask label loading
Add robustness evaluation template
```

Avoid vague commits such as:

```text
update
fix stuff
changes
```

## 5. Pull Request Checklist

Before merging a contribution, verify:

- [ ] The code runs without breaking existing training scripts.
- [ ] The experiment is documented.
- [ ] New metrics are explained.
- [ ] Results are saved in a reproducible format.
- [ ] No dataset files or large checkpoints are committed.
- [ ] The README or relevant documentation is updated.

## 6. Scientific Standards

Any result should be accompanied by:

- exact command used
- Git commit hash
- dataset split
- random seed
- hardware information
- model checkpoint path
- metric definitions

## 7. Reporting Negative Results

Negative results are useful. If a method fails to improve performance, document it. Explain whether the result may be due to optimization, data quality, model design, or evaluation protocol.

## 8. Priority Roadmap

Current research priorities:

1. Verify baseline training and evaluation.
2. Run U-Net baseline.
3. Run ENet-like real-time baseline.
4. Add or validate FPS benchmarking.
5. Record class-wise IoU.
6. Run hybrid-loss ablation.
7. Run robustness tests.
8. Prepare paper-style results table.
