from britekit.core.base_config import BaseConfig
from britekit.core.predictor import Predictor
from britekit.core import util


class FakeAudio:
    def __init__(self, cfg):
        self.set_config(cfg)

    def set_config(self, cfg):
        self.cfg = cfg
        self.sampling_rate = cfg.audio.sampling_rate


class FakeModel:
    def __init__(self, sampling_rate):
        training_cfg = util.cfg_to_pure(BaseConfig())
        training_cfg["audio"]["sampling_rate"] = sampling_rate
        self.training_cfg = training_cfg

    def apply_training_config(self, cfg):
        cfg.audio.sampling_rate = self.training_cfg["audio"]["sampling_rate"]
        cfg.train.sed_fps = self.training_cfg["train"]["sed_fps"]

    def set_config(self, cfg):
        self.cfg = cfg


def make_predictor(monkeypatch, cfg, sampling_rate, audio_overrides=None):
    model = FakeModel(sampling_rate)

    def load_models(predictor, _):
        predictor.models = [model]
        predictor.class_names = ["Species"]
        predictor.class_codes = ["SPEC"]
        predictor.class_alt_names = ["Species"]
        predictor.class_alt_codes = ["SPEC"]

    monkeypatch.setattr("britekit.core.audio.Audio", FakeAudio)
    monkeypatch.setattr(Predictor, "_load_models", load_models)
    predictor = Predictor(
        "unused.ckpt",
        device="cuda",
        cfg=cfg,
        audio_overrides=audio_overrides,
    )
    return predictor, model


def test_predictor_owns_config_snapshot(monkeypatch):
    cfg = BaseConfig()
    first, first_model = make_predictor(monkeypatch, cfg, 32000)
    second, second_model = make_predictor(monkeypatch, cfg, 44100)

    assert cfg.audio.sampling_rate == 18000
    assert first.cfg.audio.sampling_rate == 32000
    assert second.cfg.audio.sampling_rate == 44100
    assert first.audio.sampling_rate == 32000
    assert second.audio.sampling_rate == 44100
    assert first_model.cfg is first.cfg
    assert second_model.cfg is second.cfg
    assert first.cfg is not second.cfg


def test_explicit_audio_override_wins_over_checkpoint(monkeypatch):
    predictor, _ = make_predictor(
        monkeypatch,
        BaseConfig(),
        32000,
        audio_overrides={"sampling_rate": 22050},
    )

    assert predictor.cfg.audio.sampling_rate == 22050
    assert predictor.audio.sampling_rate == 22050
