import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional

class CamVidDataset(Dataset):
    """
    PyTorch Dataset for the CamVid dataset.

    Expected directory structure:
    data/camvid/
        train/
            leftImg8bit/
                *.png
            gtSeg8bit/
                *.png
        val/
            leftImg8bit/
                *.png
            gtSeg8bit/
                *.png
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        num_classes: int = 32
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes

        self.img_dir = os.path.join(root_dir, split, 'leftImg8bit')
        self.mask_dir = os.path.join(root_dir, split, 'gtSeg8bit')

        # Find all images and masks
        self.images = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
        self.masks = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))

        if not self.images:
            raise FileNotFoundError(f"No images found in {self.img_dir}. Please check the dataset path.")

        if len(self.images) != len(self.masks):
            raise ValueError(f"Mismatch between number of images ({len(self.images)}) and masks ({len(self.masks)}).")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        # Ensure mask values are appropriate for CrossEntropyLoss (0 to num_classes-1)
        # Note: This assumes the CamVid masks in gtSeg8bit are already single-channel
        # containing class indices.
        mask = mask.astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default minimal transform if none provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

def get_camvid_transforms(image_size: Tuple[int, int], augment: bool = True) -> A.Compose:
    """
    Returns a composition of Albumentations transforms.
    """
    if augment:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
