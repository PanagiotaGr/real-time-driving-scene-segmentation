import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.config import get_config
from src.datasets import get_camvid_dataloaders
from src.models.unet import UNet
from src.models.enet import ENet
from src.utils.losses import CombinedLoss
from src.utils.hybrid_loss import HybridSegmentationLoss
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML experiment configuration file.",
    )
    return parser.parse_args()


def initialize_model(config):
    """
    Initializes the model based on the configuration.
    """
    model_name = config.model.model_name.lower()
    num_classes = config.model.num_classes
    num_channels = config.model.input_channels

    if model_name == "unet":
        print("Initializing U-Net...")
        return UNet(n_channels=num_channels, n_classes=num_classes)

    if model_name == "enet":
        print("Initializing ENet...")
        return ENet(n_channels=num_channels, n_classes=num_classes)

    if model_name == "bisenet":
        try:
            from src.models.bisenet import BiSeNet
        except ImportError as exc:
            raise ImportError(
                "BiSeNet was selected, but src/models/bisenet.py is not available yet. "
                "Use model_name='unet' or model_name='enet', or add a BiSeNet implementation."
            ) from exc
        print("Initializing BiSeNet...")
        return BiSeNet(n_channels=num_channels, n_classes=num_classes)

    raise ValueError(f"Unknown model name: {model_name}")


def initialize_loss(config):
    """Initializes the loss function for controlled ablation experiments."""
    loss_type = config.training.loss_type.lower()

    if loss_type in {"cross_entropy", "ce"}:
        print("Using CrossEntropyLoss...")
        return nn.CrossEntropyLoss()

    if loss_type in {"combined", "ce_dice", "dice"}:
        print("Using CombinedLoss: Cross Entropy + Dice...")
        return CombinedLoss(dice_weight=config.training.weight_dice)

    if loss_type in {"hybrid", "ce_dice_boundary"}:
        print("Using HybridSegmentationLoss: Cross Entropy + Dice + Boundary...")
        return HybridSegmentationLoss(
            alpha=1.0,
            beta=config.training.weight_dice,
            gamma=0.1,
        )

    raise ValueError(f"Unknown loss type: {config.training.loss_type}")


def main():
    args = parse_args()
    config = get_config(args.config)

    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Loading datasets from {config.dataset.data_dir}...")
    train_loader, val_loader = get_camvid_dataloaders(
        config.dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )

    model = initialize_model(config)
    criterion = initialize_loss(config)

    if config.training.optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    elif config.training.optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.training.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer_type}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.training.device,
        config=config,
        num_classes=config.dataset.num_classes
    )

    history = trainer.fit()

    os.makedirs(config.experiment.output_dir, exist_ok=True)
    history_path = os.path.join(config.experiment.output_dir, f"{config.experiment.experiment_name}_history.yaml")
    with open(history_path, 'w') as f:
        yaml.dump(history, f)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
