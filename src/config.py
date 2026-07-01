import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DatasetConfig:
    data_dir: str = "data/camvid"
    image_size: Tuple[int, int] = (512, 512)
    num_classes: int = 32
    train_split: float = 0.8
    augmentations: bool = True


@dataclass
class ModelConfig:
    model_name: str = "unet"
    num_classes: int = 32
    input_channels: int = 3
    pretrained: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 50
    device: str = "cpu"
    num_workers: int = 4
    seed: int = 42
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "results/logs"
    save_best_only: bool = True
    optimizer_type: str = "Adam"
    loss_type: str = "cross_entropy"
    weight_dice: float = 0.5


@dataclass
class ExperimentConfig:
    experiment_name: str = "baseline_unet"
    output_dir: str = "results/experiments"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        try:
            import torch
            self.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.training.device = "cpu"


def get_config() -> Config:
    return Config()
