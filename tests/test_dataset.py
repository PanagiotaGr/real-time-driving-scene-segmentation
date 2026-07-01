import numpy as np
import cv2

from src.datasets.camvid import CamVidDataset, get_camvid_transforms


def test_camvid_dataset_loads_image_and_mask(tmp_path):
    root = tmp_path / "camvid"
    image_dir = root / "train" / "leftImg8bit"
    mask_dir = root / "train" / "gtSeg8bit"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 1] = 128
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1

    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(mask_dir / "sample.png"), mask)

    transform = get_camvid_transforms((32, 32), augment=False)
    dataset = CamVidDataset(str(root), split="train", transform=transform, num_classes=2)

    x, y = dataset[0]

    assert len(dataset) == 1
    assert x.shape == (3, 32, 32)
    assert y.shape == (32, 32)
