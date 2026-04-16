# Real-Time Driving Scene Semantic Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)


A comprehensive, research-oriented framework for high-performance semantic segmentation in driving environments. This project implements and compares several state-of-the-art architectures, specifically optimized for the balance between accuracy and real-time inference efficiency.

## 🎓 Academic Context

This project was designed to address the core challenges in autonomous vehicle perception: achieving high mean Intersection over Union (mIoU) while maintaining low latency for real-time processing. The implementations follow modular software engineering principles, making it suitable for inclusion in a university thesis or computer vision research paper.

### Key Research Objectives:
- **Efficiency vs. Accuracy Trade-off**: Comparative analysis of lightweight (ENet) vs. context-aware (BiSeNet) vs. baseline (U-Net) architectures.
- **Real-Time Feasibility**: Benchmarking inference latency on standardized driving datasets.
- **Robustness**: Implementation of hybrid loss functions (Cross-Entropy + Dice) to handle class imbalance in urban driving scenes.

---

## 🏗️ Architecture Overview

The repository implements three distinct architectural approaches:

1.  **U-Net (Baseline)**: A symmetric encoder-decoder architecture with skip connections, providing a robust baseline for segmentation performance.
2.  **ENet (Efficient)**: A lightweight network utilizing depthwise separable convolutions to minimize computational overhead, targeting high-speed edge deployment.
3.  **BiSeNet (Bilateral)**: A dual-path architecture that processes spatial details and semantic context in parallel, achieving a superior balance of speed and accuracy.

---

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.8+ and a CUDA-enabled GPU for optimal performance.

```bash
# Clone the repository
git clone https://github.com/PanagiotaGr/real-time-driving-scene-segmentation
cd real-time-driving-scene-segmentation

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

The project utilizes the **CamVid dataset**. Organize your data as follows:

```text
data/camvid/
├── train/
│   ├── leftImg8bit/  # Raw RGB images
│   └── gtSeg8bit/    # Ground truth segmentation masks
└── val/
    ├── leftImg8bit/
    └── gtSeg8bit/
```

### 3. Training

Run training using an experiment configuration file:

```bash
# Training with U-Net baseline
python src/training/train.py --config experiments/exp_unet_camvid.yaml

# Training with BiSeNet (High Speed)
python src/training/train.py --config experiments/exp_bisenet_camvid.yaml
```

### 4. Evaluation

Evaluate a trained model to generate detailed performance metrics:

```bash
python src/training/evaluate.py --checkpoint results/checkpoints/best.pth --config experiments/exp_unet_camvid.yaml
```

### 5. Inference

Predict segmentation masks for single images:

```bash
python src/inference/predict.py --image path/to/your_image.png --checkpoint results/checkpoints/best.pth --config experiments/exp_unet_camvid.yaml
```

---

## 📊 Metrics & Benchmarking

Performance is evaluated using standard computer vision metrics:

- **Mean IoU (mIoU)**: Primary metric for segmentation accuracy.
- **Pixel Accuracy**: Percentage of correctly classified pixels.
- **Class-wise IoU**: Detailed breakdown to analyze performance on rare classes (e.g., pedestrians, poles).
- **Inference Latency**: Measured in seconds per batch to quantify real-time capability.
- **Model Complexity**: Reported in total and trainable parameter counts.

---

## 📁 Project Structure

```text
.
├── experiments/          # YAML configuration files for hyperparameter management
├── src/
│   ├── datasets/         # CamVid loaders and Albumentations pipeline
│   ├── models/           # U-Net, ENet, and BiSeNet implementations
│   ├── training/         # Trainer class, loss functions, and training/eval logic
│   ├── utils/            # Metrics, logging, and visualization tools
│   └── inference/        # Single-image prediction utility
├── results/              # Checkpoints, logs, and visualization outputs
└── README.md
```

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV, Albumentations, Scikit-image
- **Data Management**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Config Management**: PyYAML

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed for research purposes in autonomous vehicle perception.*
