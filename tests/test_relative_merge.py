import numpy as np
import pytest
import torch

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.core.dataset import SpectrogramDataset
from britekit.core.util import compress_spectrogram
from britekit.core.audio_util import convert_to_db


@pytest.fixture(autouse=True)
def reset_config():
    yield
    set_base_config(BaseConfig())


def dataset(power=1, convert=False, levels=(20,), stored=True, teacher=False):
    cfg = BaseConfig()
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 8
    cfg.audio.spec_duration = 2
    cfg.audio.power = power
    cfg.audio.convert_to_db = convert
    cfg.train.sed_fps = 2
    cfg.train.multi_label = True
    cfg.train.simple_merge_db = list(levels)
    cfg.train.prob_simple_merge = 1
    cfg.train.prob_fade1 = 0
    cfg.train.augmentations = []
    set_base_config(cfg)
    a = np.zeros((2, 8), np.float32)
    b = a.copy()
    a[0, :4] = 0.2
    a[0, 4:] = 1  # Outside the labeled region: must not set energy ratio.
    b[1, 4:] = 0.8
    specs = [compress_spectrogram(x) for x in [a, b]]
    return SpectrogramDataset(
        specs,
        [[0], [1]],
        2,
        segment_ids=[10, 11],
        frame_label_dict=(
            {
                10: np.array([1, 1, 0, 0], np.float32),
                11: np.array([0, 0, 1, 1], np.float32),
            }
            if stored
            else None
        ),
        teacher_targets=(
            np.array([[0.8, 0.1], [0.2, 0.9]], np.float32) if teacher else None
        ),
        teacher_frame_targets=np.full((2, 2, 4), 0.2, np.float32) if teacher else None,
    )


@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("direction", [0.1, 0.9])
def test_active_energy_ratio_and_hard_frame_union(monkeypatch, power, direction):
    ds = dataset(power=power)
    monkeypatch.setattr("britekit.core.dataset.random.randrange", lambda _: 1)
    monkeypatch.setattr("britekit.core.dataset.random.random", lambda: direction)
    item = ds[0]
    x = item["input"].numpy()
    exponent = 2 if power == 1 else 1
    # Each source occupies a separate frequency band; energy includes both bands.
    e1 = np.mean(x[0, 0, :4].astype(float) ** exponent) / 2
    e2 = np.mean(x[0, 1, 4:].astype(float) ** exponent) / 2
    assert 10 * np.log10(e1 / e2) == pytest.approx(
        20 if direction < 0.5 else -20, abs=1e-4
    )
    torch.testing.assert_close(item["segment_labels"], torch.ones(2))
    torch.testing.assert_close(
        item["frame_labels"],
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )
    assert item["mixup"]


def test_conversion_follows_merge_and_teacher_labels_keep_strength(monkeypatch):
    monkeypatch.setattr("britekit.core.dataset.random.randrange", lambda _: 1)
    monkeypatch.setattr("britekit.core.dataset.random.random", lambda: 0.1)
    linear = dataset(teacher=True)[0]
    db = dataset(convert=True, teacher=True)[0]
    np.testing.assert_allclose(
        db["input"],
        convert_to_db(linear["input"].numpy(), 1, 80),
    )
    torch.testing.assert_close(db["teacher_segment_labels"], torch.tensor([0.84, 0.91]))
    torch.testing.assert_close(db["teacher_frame_labels"], torch.full((4, 2), 0.36))


def test_missing_frames_fall_back_per_source(monkeypatch):
    ds = dataset()
    del ds.frame_label_dict[11]
    monkeypatch.setattr("britekit.core.dataset.random.randrange", lambda _: 1)
    item = ds[0]
    torch.testing.assert_close(
        item["frame_labels"][:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(item["frame_labels"][:, 1], torch.ones(4))


@pytest.mark.parametrize("silent", [False, True])
def test_skipped_merge_does_not_add_labels(monkeypatch, silent):
    ds = dataset(teacher=True)
    if silent:
        ds.compressed_specs[1] = compress_spectrogram(np.zeros((2, 8), np.float32))
    else:
        ds.class_indexes[1] = [0]
    monkeypatch.setattr("britekit.core.dataset.random.randrange", lambda _: 1)
    item = ds[0]
    assert not item["mixup"]
    torch.testing.assert_close(item["segment_labels"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(item["teacher_segment_labels"], torch.tensor([0.8, 0.1]))


@pytest.mark.parametrize("levels", [[], [-1], [float("nan")], [float("inf")]])
def test_invalid_levels(levels):
    with pytest.raises(ValueError, match="simple_merge_db"):
        dataset(levels=levels)


def test_unsupported_features():
    with pytest.raises(ValueError, match="linear"):
        dataset(power=3)


def test_default_preserves_old_merge(monkeypatch):
    ds = dataset()
    ds.cfg.train.simple_merge_db = None
    monkeypatch.setattr("britekit.core.dataset.random.randint", lambda *_: 1)
    item = ds[0]
    # Existing merge still broadcasts labels; new option alone enables temporal union.
    torch.testing.assert_close(item["frame_labels"], torch.ones((4, 2)))
