import torch
import torch.nn as nn
import torch.optim as optim

from src.models.unet import UNet


def test_single_training_step_runs():
    model = UNet(n_channels=3, n_classes=3, features=[8, 16, 32, 64])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, 3, (2, 64, 64))

    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()

    assert outputs.shape == (2, 3, 64, 64)
    assert torch.isfinite(loss)
