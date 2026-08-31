#!/usr/bin/env python3

import os
from pathlib import Path

import librosa
import numpy as np

from britekit.core import audio, plot
from britekit.core.config_loader import get_config

recording_path = str(Path("tests") / "recordings")
plot_path = str(Path("tests") / "plots")
os.makedirs(plot_path, exist_ok=True)


def test_audio_main():
    cfg = get_config()
    _audio = audio.Audio()
    _audio.load(str(Path(recording_path) / "CommonYellowthroat.mp3"))

    specs, _ = _audio.get_spectrograms([0])
    assert len(specs) == 1
    assert specs[0].shape == (cfg.audio.spec_height, cfg.audio.spec_width)

    # plot it to allow manual check
    plot.plot_spec(
        specs[0], str(Path(plot_path) / "CommonYellowthroat.jpeg"), show_dims=True
    )


def test_low_band():
    cfg = get_config()

    # have to set audio parameters before creating audio object
    cfg.audio.spec_height = 64
    cfg.audio.freq_scale = "linear"
    cfg.audio.min_freq = 20
    cfg.audio.max_freq = 200
    _audio = audio.Audio()

    _audio.load(str(Path(recording_path) / "RuffedGrouse.mp3"))
    specs, _ = _audio.get_spectrograms([0])
    assert len(specs) == 1
    assert specs[0].shape == (cfg.audio.spec_height, cfg.audio.spec_width)

    # plot it to allow manual check
    plot.plot_spec(specs[0], str(Path(plot_path) / "RuffedGrouse.jpeg"), show_dims=True)


def test_load_records_errors_and_clears_stale_error(monkeypatch):
    _audio = audio.Audio()

    def fail_load(*args, **kwargs):
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(librosa, "load", fail_load)
    signal, _ = _audio.load("broken.mp3")
    assert signal is None
    assert _audio.load_error == "decoder unavailable"

    monkeypatch.setattr(
        librosa,
        "load",
        lambda *args, **kwargs: (
            np.zeros(_audio.cfg.audio.sampling_rate, dtype=np.float32),
            _audio.cfg.audio.sampling_rate,
        ),
    )
    signal, _ = _audio.load("working.wav")
    assert signal is not None
    assert _audio.load_error is None


def test_load_records_invalid_path_error():
    _audio = audio.Audio()

    signal, _ = _audio.load("")

    assert signal is None
    assert _audio.load_error == "Invalid path provided: "
