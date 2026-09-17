from unittest.mock import Mock

import numpy as np
import pytest

from britekit.core.audio_util import convert_to_db
from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.core.dataset import SpectrogramDataset
from britekit.core.util import compress_spectrogram


@pytest.fixture(autouse=True)
def reset_config():
    yield
    set_base_config(BaseConfig())


@pytest.mark.parametrize("normalization", ["magnitude", "minmax"])
@pytest.mark.parametrize(
    "mode", ["train", "validation", "no_augmentation", "skip_fade"]
)
def test_final_fade_order_and_training_scope(monkeypatch, normalization, mode):
    cfg = BaseConfig()
    cfg.audio.spec_height = cfg.audio.spec_width = 2
    cfg.audio.convert_to_db = normalization != "magnitude"
    cfg.audio.db_power = 1.7 if normalization == "minmax" else 1
    cfg.train.multi_label = False
    cfg.train.augmentations = []
    cfg.train.augment = mode != "no_augmentation"
    cfg.train.prob_fade1 = 0.5
    cfg.train.min_fade1 = 0.2
    cfg.train.max_fade1 = 0.4
    set_base_config(cfg)
    monkeypatch.setattr(
        "britekit.core.augmentation.random.random",
        lambda: 0.9 if mode == "skip_fade" else 0.1,
    )
    gain = Mock(return_value=0.25)
    monkeypatch.setattr("britekit.core.augmentation.random.uniform", gain)
    values = np.array([[0, 0.01], [0.2, 1]], dtype=np.float32)
    ds = SpectrogramDataset(
        [compress_spectrogram(values, bits=16)],
        [[0]],
        1,
        is_training=mode != "validation",
    )
    source = ds._get_spec(0)
    fading = mode == "train"
    if normalization == "magnitude":
        expected = source * (0.25 if fading else 1)
    else:
        expected = convert_to_db(source, db_power=cfg.audio.db_power) * (
            0.25 if fading else 1
        )
    item = ds[0]
    np.testing.assert_allclose(item["input"].numpy(), expected, atol=1e-6)
    np.testing.assert_array_equal(item["segment_labels"].numpy(), [1])
    if fading:
        gain.assert_called_once_with(0.2, 0.4)
    else:
        gain.assert_not_called()
