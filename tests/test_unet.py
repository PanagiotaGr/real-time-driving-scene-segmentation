import torch

from src.models.unet import UNet


def test_unet_forward_shape():
    model = UNet(n_channels=3, n_classes=5, features=[8, 16, 32, 64])
    model.eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 5, 64, 64)
