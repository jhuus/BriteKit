#!/usr/bin/env python3

from unittest.mock import patch

import numpy as np
import pytest

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.core.dataset import SpectrogramDataset
from britekit.core.util import compress_spectrogram


@pytest.fixture
def frame_label_cfg():
    cfg = BaseConfig()
    cfg.audio.spec_height = 4
    cfg.audio.spec_width = 12
    cfg.audio.spec_duration = 3.0
    cfg.train.sed_fps = 2
    cfg.train.augment = True
    cfg.train.max_augmentations = 1
    cfg.train.multi_label = False
    cfg.train.prob_fade1 = 0.0
    set_base_config(cfg)
    yield cfg
    set_base_config(BaseConfig())


def _dataset(frame_label_cfg, frame_labels):
    spec = np.ones(
        (frame_label_cfg.audio.spec_height, frame_label_cfg.audio.spec_width),
        dtype=np.float32,
    )
    return SpectrogramDataset(
        compressed_specs=[compress_spectrogram(spec)],
        class_indexes=[[0]],
        num_classes=1,
        segment_ids=[101],
        frame_label_dict={101: np.asarray(frame_labels, dtype=np.float32)},
    )


def test_frame_labels_roll_with_shift_horizontal(frame_label_cfg):
    frame_label_cfg.train.augmentations = [
        {"name": "shift_horizontal", "prob": 1.0, "params": {"max_shift": 2}}
    ]
    ds = _dataset(frame_label_cfg, [0, 1, 1, 0, 0, 0])

    with (
        patch("britekit.core.augmentation.random.random", return_value=0.0),
        patch("britekit.core.augmentation.random.randint", return_value=2),
    ):
        item = ds[0]

    np.testing.assert_array_equal(
        item["frame_labels"][:, 0].numpy(),
        np.array([0, 0, 1, 1, 0, 0], dtype=np.float32),
    )


def test_frame_labels_shift_with_padding_for_shift_horizontal(frame_label_cfg):
    frame_label_cfg.train.augmentations = [
        {
            "name": "shift_horizontal",
            "prob": 1.0,
            "params": {"max_shift": 2, "pad_value": 0.0},
        }
    ]
    ds = _dataset(frame_label_cfg, [1, 1, 0, 0, 0, 0])

    with (
        patch("britekit.core.augmentation.random.random", return_value=0.0),
        patch("britekit.core.augmentation.random.randint", return_value=2),
    ):
        item = ds[0]

    np.testing.assert_array_equal(
        item["frame_labels"][:, 0].numpy(),
        np.array([0, 1, 1, 0, 0, 0], dtype=np.float32),
    )


def test_frame_labels_are_zeroed_for_time_mask(frame_label_cfg):
    frame_label_cfg.train.augmentations = [
        {"name": "time_mask", "prob": 1.0, "params": {"max_width2": 4}}
    ]
    ds = _dataset(frame_label_cfg, [1, 1, 1, 1, 1, 1])

    with (
        patch("britekit.core.augmentation.random.random", return_value=0.0),
        patch("britekit.core.augmentation.np.random.randint", side_effect=[4, 4]),
    ):
        item = ds[0]

    np.testing.assert_array_equal(
        item["frame_labels"][:, 0].numpy(),
        np.array([1, 1, 0, 0, 1, 1], dtype=np.float32),
    )
