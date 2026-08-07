from types import SimpleNamespace

import numpy as np

from britekit.commands._plot import _plot_recording


class FakeAudio:
    def __init__(self, recording_seconds: float):
        self.signal = np.zeros(round(recording_seconds * 10), dtype=np.float32)
        self.skip_cache = None

    def load(self, _path: str):
        return self.signal, 10

    def get_spectrograms(self, _offsets, spec_duration, skip_cache):
        self.skip_cache = skip_cache
        return np.zeros((1, 4, 4), dtype=np.float32), None


def test_plot_entire_short_recording_uses_minimum_width(monkeypatch, tmp_path):
    plotted = {}

    def capture_plot(_spec, _path, **kwargs):
        plotted.update(kwargs)

    monkeypatch.setattr("britekit.core.plot.plot_spec", capture_plot)
    cfg = SimpleNamespace(audio=SimpleNamespace(spec_width=480, spec_duration=5.0))

    audio = FakeAudio(recording_seconds=0.6)
    _plot_recording(
        cfg,
        audio,
        "short.mp3",
        str(tmp_path),
        all=True,
        overlap=0,
        ndims=False,
    )

    assert plotted["width"] == 640
    assert plotted["height"] == 300
    assert plotted["spec_duration"] == 0.6
    assert audio.skip_cache is True


def test_plot_entire_recording_handles_empty_spectrograms(
    monkeypatch, tmp_path, caplog
):
    audio = FakeAudio(recording_seconds=0.4)
    monkeypatch.setattr(
        audio,
        "get_spectrograms",
        lambda *_args, **_kwargs: (np.empty(0), np.empty(0)),
    )

    _plot_recording(
        SimpleNamespace(audio=SimpleNamespace(spec_width=480, spec_duration=5.0)),
        audio,
        "too-short.mp3",
        str(tmp_path),
        all=True,
        overlap=0,
        ndims=False,
    )

    assert "failed to extract spectrogram" in caplog.text
