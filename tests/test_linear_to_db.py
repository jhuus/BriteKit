from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from britekit.core.audio import Audio
from britekit.core.audio_util import convert_to_db
from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.core.dataset import SpectrogramDataset
from britekit.core.predictor import Predictor
from britekit.core.util import compress_spectrogram, cfg_to_pure


@pytest.mark.parametrize("power", [1, 2])
def test_relative_db_conversion_is_per_segment_and_gain_invariant(power):
    x = np.array([[[0, 1e-3], [0.1, 1]], [[0, 0], [0, 0]]], dtype=np.float32)
    result = convert_to_db(x, power, 80)
    expected = np.array([[0, 1 - (20 / power) * 3 / 80], [1 - (20 / power) / 80, 1]])
    np.testing.assert_allclose(result[0], expected, atol=1e-6)
    np.testing.assert_array_equal(result[1], 0)
    np.testing.assert_allclose(convert_to_db(x * 0.15, power), result, atol=1e-6)
    np.testing.assert_allclose(x[0], [[0, 1e-3], [0.1, 1]])


def audio_fixture():
    cfg = BaseConfig()
    cfg.audio.spec_height = 8
    cfg.audio.spec_width = 32
    cfg.audio.spec_duration = 1
    cfg.audio.use_spec_cache = True
    a = Audio(device="cpu", cfg=cfg)
    a.signal = (
        np.random.default_rng(1)
        .normal(0, 0.05, 4 * cfg.audio.sampling_rate)
        .astype(np.float32)
    )
    return a


def test_on_demand_conversion_reuses_only_linear_cache(
    monkeypatch,
):
    a = audio_fixture()
    raw_call = Mock(wraps=a._get_raw_spectrogram)
    monkeypatch.setattr(a, "_get_raw_spectrogram", raw_call)
    linear, _ = a.get_spectrograms([0, 1], convert_to_db=False)
    converted, raw_db = a.get_spectrograms([0, 1], convert_to_db=True)
    np.testing.assert_allclose(converted, convert_to_db(linear), atol=1e-6)
    assert raw_call.call_count == 1
    assert raw_db.max() == 0
    assert raw_db.min() >= -80
    # Every request converts again but does not recompute the linear spectrogram.
    reordered, _ = a.get_spectrograms([1, 0], convert_to_db=True)
    np.testing.assert_array_equal(reordered, converted[::-1])
    assert raw_call.call_count == 1
    reordered[:] = 42
    again, _ = a.get_spectrograms([0], convert_to_db=True)
    np.testing.assert_array_equal(again[0], converted[0])
    a.set_config(a.cfg)
    assert a.cached is None


def test_transform_cache_respects_power():
    a = audio_fixture()
    a.cfg.audio.power = 2
    a.set_config(a.cfg)
    assert a.mel_transform.spectrogram.power == 2
    a.cfg.audio.power = 1
    a.set_config(a.cfg)
    assert a.mel_transform.spectrogram.power == 1


def test_training_converts_after_augmentation_and_validation_uses_same_conversion():
    cfg = BaseConfig()
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 2
    cfg.audio.convert_to_db = True
    cfg.train.augment = False
    cfg.train.prob_fade1 = 0
    set_base_config(cfg)
    try:
        x = np.array([[0, 0.2], [0.5, 1]], dtype=np.float32)
        ds = SpectrogramDataset([compress_spectrogram(x)], [[0]], 1)
        source = ds._get_spec(0)
        np.testing.assert_allclose(ds[0]["input"], convert_to_db(source))
        ds.augment = lambda spec, frame_labels: (spec + 0.1, frame_labels)
        cfg.train.multi_label = False
        np.testing.assert_allclose(ds[0]["input"], convert_to_db(source + 0.1))
        ds.is_training = False
        np.testing.assert_allclose(ds[0]["input"], convert_to_db(source))
    finally:
        set_base_config(BaseConfig())


class Model:
    def __init__(self, cfg):
        self.training_cfg = cfg_to_pure(cfg)
        self.seen = None

    def apply_training_config(self, cfg):
        pass

    def set_config(self, cfg):
        self.cfg = cfg

    def predict(self, specs, device):
        self.seen = specs.copy()
        return specs.mean(axis=(1, 2, 3))[:, None], None


def mixed_predictor():
    cfg = BaseConfig()
    db = deepcopy(cfg)
    db.audio.convert_to_db = True
    predictor = Predictor.__new__(Predictor)
    predictor.cfg = cfg
    predictor.audio_overrides = {}
    predictor.models = [Model(cfg), Model(db)]
    predictor.ov = None
    predictor.device = "cpu"
    predictor._apply_model_config()
    return predictor


def test_mixed_ensemble_routes_conversion_before_inference_exponent():
    p = mixed_predictor()
    x = np.array([[[[0, 0.01], [0.1, 1]]]], dtype=np.float32)
    p.get_block_scores(x, audio_power=0.7)
    np.testing.assert_allclose(p.models[0].seen, x**0.7)
    np.testing.assert_allclose(p.models[1].seen, convert_to_db(x) ** 0.7)
    np.testing.assert_array_equal(x[0, 0, 0, 0], 0)


def test_mixed_ensemble_rejects_incompatible_linear_features():
    p = mixed_predictor()
    p.models[1].training_cfg["audio"]["power"] = 2
    with pytest.raises(Exception, match="linear features differ: power"):
        p._apply_model_config()


def test_mixed_overlapping_inference_converts_requested_windows(tmp_path):
    p = mixed_predictor()
    p.cfg.audio = audio_fixture().cfg.audio
    p.audio = audio_fixture()
    p.audio.load = lambda path: None
    p.cfg.audio.spec_duration = 3
    p.audio.set_config(p.cfg)
    recording = tmp_path / "audio.wav"
    recording.touch()
    result = p.get_overlapping_scores(str(recording), initial_start_times=[0, 1.5])
    assert result is not None
    for i, model in enumerate(p.models):
        starts = p.get_start_times(p.audio.seconds(), [0, 1.5][i], 3, overlap=0)
        if i:
            starts = [-1.5] + starts
        linear, _ = p.audio.get_spectrograms(starts, convert_to_db=False)
        expected = convert_to_db(linear) if i else linear
        np.testing.assert_allclose(
            model.seen[:, 0], expected**p.cfg.infer.audio_power, atol=1e-6
        )


def test_new_mode_rejects_legacy_db():
    a = audio_fixture()
    with pytest.raises(ValueError, match="decibels=False"):
        a.get_spectrograms([0], decibels=True, convert_to_db=True)


@pytest.mark.parametrize("value", [np.nan, -1, np.inf])
def test_conversion_rejects_invalid_linear_values(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        convert_to_db(np.array([[0, value]], dtype=np.float32))


def test_audio_uses_shared_minmax_mapping(monkeypatch):
    a = audio_fixture()
    linear = np.full((1, 8, 32), 0.01, dtype=np.float32)
    linear[..., -1] = 1
    # Supply a dense normalized linear image to distinguish the two mappings.
    calls = []

    def prepare(values):
        calls.append(True)
        values[:] = linear

    monkeypatch.setattr(a, "_normalize", prepare)
    output, raw = a.get_spectrograms([0], convert_to_db=True)
    assert len(calls) == 1
    np.testing.assert_allclose(output, convert_to_db(linear))
    np.testing.assert_allclose(raw, convert_to_db(linear, normalize=False))
    output, _ = a.get_spectrograms([0], convert_to_db=True)
    np.testing.assert_allclose(output, convert_to_db(linear))


def test_dataset_conversion_runs_after_augmentation():
    cfg = BaseConfig()
    cfg.audio.spec_height = cfg.audio.spec_width = 2
    cfg.audio.convert_to_db = True
    cfg.train.augment = False
    cfg.train.prob_fade1 = 0
    cfg.train.multi_label = False
    set_base_config(cfg)
    try:
        x = np.array([[0, 0.2], [0.5, 1]], dtype=np.float32)
        ds = SpectrogramDataset([compress_spectrogram(x)], [[0]], 1)
        source = ds._get_spec(0)
        ds.augment = lambda spec, frame_labels: (spec + 0.1, frame_labels)
        expected = convert_to_db(source + 0.1)
        np.testing.assert_allclose(ds[0]["input"], expected)
        assert expected.min() == 0
        ds.is_training = False
        np.testing.assert_allclose(ds[0]["input"], convert_to_db(source))
    finally:
        set_base_config(BaseConfig())


@pytest.mark.parametrize("exponent", [0.7, 1.0, 1.3, 1.6])
def test_db_power_applied_after_normalization(exponent):
    x = np.array([[[0, 1e-4], [0.1, 1]], [[0, 0], [0, 0]]], dtype=np.float32)
    original = x.copy()
    baseline = convert_to_db(x)
    output = convert_to_db(x, db_power=exponent)
    np.testing.assert_allclose(output, baseline**exponent)
    np.testing.assert_array_equal(x, original)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert output.min() == 0 and output.max() == 1
    np.testing.assert_array_equal(
        convert_to_db(x, normalize=False, db_power=exponent),
        convert_to_db(x, normalize=False),
    )


@pytest.mark.parametrize("exponent", [0, -1, np.nan, np.inf])
def test_invalid_converted_db_power_rejected(exponent):
    from britekit.core.config_loader import get_config

    with pytest.raises(ValueError, match="db_power"):
        convert_to_db(np.ones((2, 2)), db_power=exponent)
    cfg = BaseConfig()
    cfg.audio.convert_to_db = True
    cfg.audio.db_power = exponent
    set_base_config(cfg)
    try:
        with pytest.raises(ValueError, match="db_power"):
            get_config()
    finally:
        set_base_config(BaseConfig())


def test_db_power_audio_cache_and_override():
    a = audio_fixture()
    a.cfg.audio.db_power = 1.3
    linear, _ = a.get_spectrograms([0], convert_to_db=False)
    cached = a.cached
    for exponent in (None, 0.7, 1.6):
        output, raw = a.get_spectrograms([0], convert_to_db=True, db_power=exponent)
        assert a.cached is cached
        np.testing.assert_allclose(
            output, convert_to_db(linear) ** (1.3 if exponent is None else exponent)
        )
        np.testing.assert_array_equal(raw, convert_to_db(linear, normalize=False))


def test_db_power_dataset_after_augmentation_with_relative_merge_enabled():
    cfg = BaseConfig()
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 2
    cfg.audio.convert_to_db = True
    cfg.audio.db_power = 1.3
    cfg.train.simple_merge_db = [20]
    cfg.train.prob_simple_merge = 0
    cfg.train.augment = False
    cfg.train.prob_fade1 = 0
    set_base_config(cfg)
    try:
        values = np.array([[0, 0.01], [0.2, 1]], dtype=np.float32)
        ds = SpectrogramDataset([compress_spectrogram(values)], [[0]], 1)
        source = ds._get_spec(0)
        np.testing.assert_allclose(ds[0]["input"], convert_to_db(source) ** 1.3)
        cfg.train.multi_label = False
        ds.augment = lambda spec, frame_labels: (spec + 0.1, frame_labels)
        np.testing.assert_allclose(
            ds[0]["input"],
            convert_to_db(source + 0.1) ** 1.3,
        )
    finally:
        set_base_config(BaseConfig())


def test_mixed_db_powers_are_selected_per_checkpoint_and_cached_separately():
    p = mixed_predictor()
    for model, exponent in zip(p.models, [0.7, 1.6]):
        model.training_cfg["audio"]["convert_to_db"] = True
        model.training_cfg["audio"]["db_power"] = exponent
    p._apply_model_config()
    x = np.array([[[[0, 0.01], [0.1, 1]]]], dtype=np.float32)
    p.get_block_scores(x, audio_power=1)
    for model, exponent in zip(p.models, [0.7, 1.6]):
        np.testing.assert_allclose(model.seen, convert_to_db(x) ** exponent)
    del p.models[0].training_cfg["audio"]["db_power"]
    p._apply_model_config()
    assert p.model_audio_settings[0]["db_power"] == 1


def test_overlapping_inference_uses_per_checkpoint_db_power(tmp_path):
    p = mixed_predictor()
    for model, exponent in zip(p.models, [0.7, 1.6]):
        model.training_cfg["audio"]["convert_to_db"] = True
        model.training_cfg["audio"]["db_power"] = exponent
    p._apply_model_config()
    p.cfg.audio = audio_fixture().cfg.audio
    p.cfg.audio.db_power = 2  # Must not override each checkpoint's exponent.
    p.cfg.infer.audio_power = 1
    p.audio = audio_fixture()
    p.audio.load = lambda path: None
    p.cfg.audio.spec_duration = 3
    p.audio.set_config(p.cfg)
    recording = tmp_path / "audio.wav"
    recording.touch()
    p.get_overlapping_scores(str(recording), initial_start_times=[0, 1.5])
    for i, (model, exponent) in enumerate(zip(p.models, [0.7, 1.6])):
        starts = p.get_start_times(p.audio.seconds(), [0, 1.5][i], 3, overlap=0)
        if i:
            starts = [-1.5] + starts
        linear, _ = p.audio.get_spectrograms(starts, convert_to_db=False)
        np.testing.assert_allclose(
            model.seen[:, 0], convert_to_db(linear) ** exponent, atol=1e-6
        )
