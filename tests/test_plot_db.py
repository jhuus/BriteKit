from types import SimpleNamespace

import numpy as np
import pytest

from britekit.commands._plot import plot_db
from britekit.core.audio_util import convert_to_db
from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config


@pytest.mark.parametrize("convert", [False, True])
@pytest.mark.parametrize("augment", [False, True])
def test_plot_db_conversion_order(monkeypatch, tmp_path, convert, augment):
    cfg = BaseConfig()
    cfg.audio.convert_to_db = convert
    cfg.audio.db_power = 1.3
    cfg.audio.top_db = 80
    cfg.audio.power = 1
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 2
    source = np.array([[0, 0.001], [0.1, 1]], dtype=np.float32)
    augmented = np.array([[0.001, 0.01], [0.2, 1]], dtype=np.float32)
    captured = []

    class FakeDatabase:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def get_spectrogram_by_class(self, *_args, **_kwargs):
            return [SimpleNamespace(filename="example.wav", offset=0, value=b"")]

    def fake_augment(spec):
        np.testing.assert_array_equal(spec, source[None])
        return augmented[None].copy()

    monkeypatch.setattr("britekit.commands._plot.get_config", lambda _: cfg)
    monkeypatch.setattr(
        "britekit.training_db.training_db.TrainingDatabase", FakeDatabase
    )
    monkeypatch.setattr(
        "britekit.commands._plot.util.expand_spectrogram", lambda _: source.copy()
    )
    monkeypatch.setattr(
        "britekit.core.augmentation.AugmentationPipeline", lambda *_: fake_augment
    )
    monkeypatch.setattr(
        "britekit.core.plot.plot_spec",
        lambda spec, *_args, **_kwargs: captured.append(spec.copy()),
    )
    exponent = 0.4
    plot_db(
        class_name="BCCH", output_path=str(tmp_path), power=exponent, augment=augment
    )
    expected = augmented if augment else source
    if convert:
        expected = convert_to_db(expected, 1, 80, db_power=cfg.audio.db_power)
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], expected**exponent)


def test_plot_db_cli_forwards_fixed_limits(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from britekit.commands._plot import _plot_db_cmd

    calls = []
    monkeypatch.setattr(
        "britekit.commands._plot.plot_db", lambda *args: calls.append(args)
    )
    result = CliRunner().invoke(
        _plot_db_cmd,
        ["--name", "Bird", "-o", str(tmp_path), "--vmin", "-7", "--vmax", "2"],
    )
    assert result.exit_code == 0, result.output
    assert calls[0][-2:] == (-7, 2)


def test_plot_uses_fixed_color_limits(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from britekit.core.plot import plot_spec

    set_base_config(BaseConfig())
    captured = []
    original = plt.pcolormesh

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result.get_clim())
        return result

    monkeypatch.setattr(plt, "pcolormesh", capture)
    output = tmp_path / "fixed_limits.png"
    try:
        plot_spec(np.array([[-4, -1], [0, 1]]), str(output), vmin=-7, vmax=2)
        assert captured == [(-7, 2)]
        assert output.is_file()
    finally:
        plt.close("all")
        set_base_config(BaseConfig())
