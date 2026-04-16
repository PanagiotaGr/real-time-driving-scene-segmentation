import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
from src.config import get_config
from src.datasets import get_camvid_dataloaders
from src.models.unet import UNet
from src.models.enet import ENet
from src.models.bisenet import BiSeNet
from src.utils.losses import CombinedLoss
from src.training.trainer import Trainer

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
    elif model_name == "enet":
        print("Initializing ENet...")
        return ENet(n_channels=num_channels, n_classes=num_classes)
    elif model_name == "bisenet":
        print("Initializing BiSeNet...")
        return BiSeNet(n_channels=num_channels, n_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    # 1. Configuration
    config = get_config()

    # Reproducibility
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2. Data
    print(f"Loading datasets from {config.dataset.data_dir}...")
    train_loader, val_loader = get_camvid_dataloaders(
        config.dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )

    # 3. Model
    model = initialize_model(config)

    # 4. Loss and Optimizer
    # Using CombinedLoss (CE + Dice) as it's better for segmentation
    criterion = CombinedLoss(dice_weight=config.training.weight_dice)

    if config.training.optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    elif config.training.optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.training.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer_type}")

    # 5. Trainer
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

    # 6. Execution
    history = trainer.fit()

    # 7. Save History
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    history_path = os.path.join(config.experiment.output_dir, f"{config.experiment.experiment_name}_history.yaml")
    with open(history_path, 'w') as f:
        yaml.dump(history, f)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()
