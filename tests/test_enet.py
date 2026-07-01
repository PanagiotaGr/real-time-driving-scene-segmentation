import torch

from src.models.enet import ENet


def test_enet_forward_shape():
    model = ENet(n_channels=3, n_classes=5)
    model.eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 5, 64, 64)
