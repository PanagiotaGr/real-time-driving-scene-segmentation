from .camvid import CamVidDataset, get_camvid_transforms
from torch.utils.data import DataLoader
from src.config import DatasetConfig

def get_camvid_dataloaders(config: DatasetConfig, batch_size: int, num_workers: int):
    """
    Creates training and validation DataLoaders for the CamVid dataset.
    """
    train_transform = get_camvid_transforms(config.image_size, augment=True)
    val_transform = get_camvid_transforms(config.image_size, augment=False)

    train_dataset = CamVidDataset(
        root_dir=config.data_dir,
        split='train',
        transform=train_transform,
        num_classes=config.num_classes
    )

    val_dataset = CamVidDataset(
        root_dir=config.data_dir,
        split='val',
        transform=val_transform,
        num_classes=config.num_classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
