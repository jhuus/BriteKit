from unittest.mock import Mock

import numpy as np
import pytest

from britekit.core.audio import Audio
from britekit.core.base_config import BaseConfig
from britekit.core.predictor import Predictor


def cached_audio():
    cfg = BaseConfig()
    cfg.audio.spec_duration = 3
    cfg.audio.spec_width = 12
    cfg.audio.spec_height = 2
    cfg.audio.use_spec_cache = True
    audio = Audio.__new__(Audio)
    audio.cfg = cfg
    audio.signal = np.ones(6 * cfg.audio.sampling_rate, dtype=np.float32)
    audio.cached = np.arange(1, 49, dtype=np.float32).reshape(2, 24)
    audio.cached_freq_scale = cfg.audio.freq_scale
    audio.load = Mock()
    return audio


def test_half_second_leading_window_is_padded_without_losing_later_windows():
    audio = cached_audio()
    normalized, raw = audio.get_spectrograms([-2.5, 0.5, 3.5])
    assert raw.shape == (3, 2, 12)
    np.testing.assert_array_equal(raw[0, :, :10], 0)
    np.testing.assert_array_equal(raw[0, :, 10:], audio.cached[:, :2])
    np.testing.assert_array_equal(raw[1], audio.cached[:, 2:14])
    np.testing.assert_array_equal(raw[2, :, :10], audio.cached[:, 14:24])
    np.testing.assert_array_equal(raw[2, :, 10:], 0)
    assert np.isfinite(normalized).all()


def test_short_trailing_window_is_still_rejected():
    audio = cached_audio()
    _, raw = audio.get_spectrograms([0, 3, 5.5])
    assert raw.shape == (2, 2, 12)


@pytest.mark.parametrize("count", [3, 6, 12])
def test_every_offset_model_contributes_to_overlapping_ensemble(count, tmp_path):
    audio = cached_audio()
    cfg = audio.cfg
    cfg.infer.audio_power = 1
    cfg.train.sed_fps = 4
    predictor = Predictor.__new__(Predictor)
    predictor.cfg = cfg
    predictor.audio = audio
    predictor.device = "cpu"
    predictor.ov = None
    predictor.models = []
    for i in range(count):
        score = (i + 1) / (count + 1)

        def predict(specs, device, score=score):
            return (
                np.full((len(specs), 1), score, dtype=np.float32),
                np.full((len(specs), 1, 12), score, dtype=np.float32),
            )

        model = Mock()
        model.predict.side_effect = predict
        predictor.models.append(model)
    path = tmp_path / "recording.wav"
    path.touch()
    frames = predictor.get_overlapping_scores(str(path), [0, 0.5, 1, 1.5, 2, 2.5])
    for model in predictor.models:
        model.predict.assert_called_once()
    # All models predict throughout the first window, including at time zero.
    np.testing.assert_allclose(frames[:12, 0], 0.5, atol=1e-6)
    # The half-second-offset member receives its leading window AND later ones.
    assert predictor.models[1].predict.call_args.args[0].shape[0] == 3
