import pytest

from src.config import get_config


def test_default_config_loads():
    config = get_config()

    assert config.dataset.data_dir == "data/camvid"
    assert config.model.model_name == "unet"
    assert config.training.batch_size == 8


def test_yaml_config_override(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
dataset:
  image_size: [64, 64]
  num_classes: 5
model:
  model_name: enet
  num_classes: 5
training:
  batch_size: 2
  epochs: 1
experiment:
  experiment_name: test_run
""",
        encoding="utf-8",
    )

    config = get_config(str(config_path))

    assert config.dataset.image_size == (64, 64)
    assert config.dataset.num_classes == 5
    assert config.model.model_name == "enet"
    assert config.training.batch_size == 2
    assert config.experiment.experiment_name == "test_run"


def test_unknown_config_field_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("training:\n  unknown_field: 123\n", encoding="utf-8")

    with pytest.raises(ValueError):
        get_config(str(config_path))
