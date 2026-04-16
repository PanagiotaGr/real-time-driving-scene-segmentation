import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from src.config import get_config
from src.datasets import get_camvid_dataloaders
from src.models.unet import UNet
from src.models.enet import ENet
from src.models.bisenet import BiSeNet
from src.utils.losses import CombinedLoss
from src.utils.metrics import SegmentationMetrics

def initialize_model(config):
    model_name = config.model.model_name.lower()
    num_classes = config.model.num_classes
    num_channels = config.model.input_channels

    if model_name == "unet":
        return UNet(n_channels=num_channels, n_classes=num_classes)
    elif model_name == "enet":
        return ENet(n_channels=num_channels, n_classes=num_classes)
    elif model_name == "bisenet":
        return BiSeNet(n_channels=num_channels, n_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

@torch.no_grad()
def evaluate(config, checkpoint_path):
    # 1. Load Configuration
    config = get_config()

    # 2. Load Model
    device = config.training.device
    model = initialize_model(config)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 3. Load Data
    _, val_loader = get_camvid_dataloaders(
        config.dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )

    # 4. Metrics & Performance
    metrics = SegmentationMetrics(num_classes=config.dataset.num_classes)

    total_time = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    print("Starting evaluation...")

    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        start_time = time.time()
        outputs = model(images)
        total_time += (time.time() - start_time)
        num_batches += 1

        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    # Calculate Metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    iou_res = metrics.iou_metrics(all_preds, all_targets)
    pixel_acc = metrics.pixel_accuracy(all_preds, all_targets)

    avg_inference_time = total_time / num_batches

    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Mean IoU:          {iou_res['mIoU']:.4f}")
    print(f"Pixel Accuracy:    {pixel_acc:.4f}")
    print(f"Avg Inference Time: {avg_inference_time:.4f} s/batch")
    print(f"Total Parameters:  {total_params:,}")
    print(f"Trainable Params:  {trainable_params:,}")
    print("="*30)

    # 5. Visualization
    visualize_results(model, val_loader, device, config, num_classes=config.dataset.num_classes)

def visualize_results(model, loader, device, config, num_classes):
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    # Pick first image in batch
    img = images[0].cpu().permute(1, 2, 0).numpy()
    # Denormalize (approximate)
    img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)

    gt = masks[0].cpu().numpy()
    pred = preds[0].cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gt)
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred)
    plt.title("Prediction")
    plt.axis('off')

    save_path = os.path.join(config.experiment.output_dir, "eval_visualization.png")
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Example: Evaluate the best model
    import sys
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    else:
        # Default to best.pth if no argument provided
        ckpt = os.path.join("semantic-segmentation-driving/results/checkpoints/best.pth")

    if os.path.exists(ckpt):
        evaluate(get_config(), ckpt)
    else:
        print(f"Checkpoint not found: {ckpt}")
