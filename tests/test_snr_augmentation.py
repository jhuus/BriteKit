from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from britekit.core.augmentation import AugmentationPipeline
from britekit.core.base_config import BaseConfig


def make_pipeline(power=1, targets=(20, 10, 0)):
    cfg = BaseConfig()
    cfg.audio.power = power
    cfg.train.prob_fade1 = 0
    cfg.train.augmentations = [
        {"name": "add_noise_snr", "prob": 1.0, "params": {"snr_db": targets}}
    ]
    noise = np.zeros((1, 2, 8), dtype=np.float32)
    noise[:, 1] = 1
    dataset = SimpleNamespace(
        num_classes=2, noise_class_index=1, get_random_noise=Mock(return_value=noise)
    )
    return AugmentationPipeline(cfg, dataset), noise


def foreground():
    spec = np.zeros((1, 2, 8), dtype=np.float32)
    spec[:, 0, :4] = 0.2
    spec[:, 0, 4:] = 2
    return spec


@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("target", [-10, 0, 10, 20])
def test_requested_ratio_over_active_frames_survives_pipeline_normalization(
    power, target
):
    pipe, noise = make_pipeline(power, [target])
    spec = foreground()
    saved_spec, saved_noise = spec.copy(), noise.copy()
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    original_labels = labels.clone()
    mixed, returned_labels = pipe(spec, frame_labels=labels)
    exponent = 2 if power == 1 else 1
    # Disjoint frequency bands identify the two components after normalization.
    measured = 10 * np.log10(
        np.mean(mixed[:, 0, :4] ** exponent) / np.mean(mixed[:, 1, :4] ** exponent)
    )
    assert measured == pytest.approx(target, abs=1e-5)
    assert mixed.dtype == np.float32
    assert mixed.max() <= 1
    assert returned_labels is labels
    torch.testing.assert_close(labels, original_labels)
    np.testing.assert_array_equal(spec, saved_spec)
    np.testing.assert_array_equal(noise, saved_noise)


@pytest.mark.parametrize("labels", [None, np.zeros((2, 2)), np.array([[0, 1], [0, 1]])])
def test_missing_inactive_or_noise_only_labels_use_whole_segment(labels):
    pipe, _ = make_pipeline(targets=[10])
    result = pipe.add_noise_snr(foreground(), snr_db=[10], frame_labels=labels)
    mixed = result if labels is None else result[0]
    measured = 10 * np.log10(np.mean(mixed[:, 0] ** 2) / np.mean(mixed[:, 1] ** 2))
    assert measured == pytest.approx(10, abs=1e-5)


def test_appended_teacher_targets_do_not_expand_active_region():
    pipe, _ = make_pipeline(targets=[0])
    # Teacher outputs are positive everywhere, hard annotations only in first half.
    labels = torch.tensor([[1.0, 0.0, 0.8, 0.9], [0.0, 0.0, 0.9, 0.8]])
    mixed, returned = pipe(foreground(), frame_labels=labels)
    assert returned is labels
    np.testing.assert_allclose(mixed[:, 0, :4], mixed[:, 1, :4])


def test_active_frame_mapping_includes_overlapping_columns():
    pipe, _ = make_pipeline(targets=[0])
    spec = foreground()
    # Middle of 3 annotation frames overlaps spectrogram columns 2 through 5.
    labels = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    mixed, _ = pipe(spec, frame_labels=labels)
    measured = np.mean(mixed[:, 0, 2:6] ** 2) / np.mean(mixed[:, 1, 2:6] ** 2)
    assert measured == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("silent", ["signal", "noise", "active_noise"])
def test_silent_regions_skip_mixing_without_nan(silent):
    pipe, noise = make_pipeline()
    spec = foreground()
    if silent == "signal":
        spec[:] = 0
    elif silent == "noise":
        noise[:] = 0
    else:
        noise[..., :4] = 0
    labels = np.array([[1.0, 0.0], [0.0, 0.0]])
    mixed, returned = pipe.add_noise_snr(spec, frame_labels=labels)
    np.testing.assert_array_equal(mixed, spec)
    assert returned is labels
    assert np.isfinite(mixed).all()


def test_missing_noise_warns_once_and_preserves_input(caplog):
    pipe, _ = make_pipeline()
    pipe.dataset.get_random_noise.return_value = None
    spec = foreground()
    assert pipe.add_noise_snr(spec) is spec
    assert pipe.add_noise_snr(spec) is spec
    assert caplog.text.count("no noise spectrograms available") == 1


@pytest.mark.parametrize("targets", [[], [np.nan], [np.inf], 10, "20"])
def test_invalid_targets_rejected_when_pipeline_is_built(targets):
    with pytest.raises(ValueError, match="snr_db"):
        make_pipeline(targets=targets)


def test_legacy_db_rejected_but_post_augmentation_conversion_allowed():
    pipe, _ = make_pipeline()
    pipe.cfg.audio.convert_to_db = True
    AugmentationPipeline(pipe.cfg, pipe.dataset)
    pipe.cfg.audio.decibels = True
    with pytest.raises(ValueError, match="linear"):
        AugmentationPipeline(pipe.cfg, pipe.dataset)


def test_yaml_target_list_and_choice(monkeypatch):
    pipe, _ = make_pipeline(targets=OmegaConf.create([20, 10, 0]))
    choice = Mock(return_value=10)
    monkeypatch.setattr("britekit.core.augmentation.random.choice", choice)
    pipe(foreground())
    assert list(choice.call_args.args[0]) == [20, 10, 0]


def test_training_dataset_passes_stored_frame_labels_to_snr_augmentation():
    from britekit.core.config_loader import set_base_config
    from britekit.core.dataset import SpectrogramDataset
    from britekit.core.util import compress_spectrogram

    pipe, noise = make_pipeline(targets=[0])
    cfg = pipe.cfg
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 8
    cfg.audio.spec_duration = 3
    cfg.train.sed_fps = 2
    cfg.train.multi_label = False
    cfg.train.augment = True
    set_base_config(cfg)
    try:
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
        ds = SpectrogramDataset(
            [compress_spectrogram(foreground()), compress_spectrogram(noise)],
            [[0], [1]],
            2,
            noise_class_index=1,
            segment_ids=[100, 200],
            frame_label_dict={100: labels},
        )
        item = ds[0]
        mixed = item["input"].numpy()
        np.testing.assert_allclose(mixed[:, 0, :4], mixed[:, 1, :4], atol=1e-6)
        np.testing.assert_array_equal(item["frame_labels"][:, 0], labels)
        np.testing.assert_array_equal(item["frame_labels"][:, 1], 0)
        np.testing.assert_array_equal(item["segment_labels"], [1, 0])
    finally:
        set_base_config(BaseConfig())
