import pickle
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from britekit.commands import _teacher_targets


class FakeModel:
    def __init__(self, class_codes, score, frame_score=None, **audio_settings):
        self.train_class_codes = class_codes
        self.score = score
        self.frame_score = frame_score
        self.audio_settings = dict(
            spec_height=2,
            spec_width=3,
            power=1.0,
            decibels=False,
            convert_to_db=False,
            top_db=80.0,
            db_power=1.0,
        )
        self.audio_settings.update(audio_settings)
        self.inputs = []

    def eval(self):
        return self

    def apply_training_config(self, cfg):
        cfg.audio = SimpleNamespace(**self.audio_settings)
        self.cfg = cfg

    def to(self, device):
        return self

    def predict(self, specs, device):
        self.inputs.append(specs.copy())
        scores = np.full(
            (len(specs), len(self.train_class_codes)), self.score, dtype=np.float32
        )
        frame_scores = None
        if self.frame_score is not None:
            frame_scores = np.full(
                (len(specs), len(self.train_class_codes), 4),
                self.frame_score,
                dtype=np.float32,
            )
        return scores, frame_scores


@pytest.fixture
def source_pickle(tmp_path):
    path = tmp_path / "training.pkl"
    data = {
        "class_codes": ["a", "b"],
        "spec_values": [np.arange(6, dtype=np.float32) for _ in range(3)],
        "spec_segment_ids": [10, 20, 30],
    }
    with path.open("wb") as file:
        pickle.dump(data, file)
    return path


def test_teacher_targets_averages_ensemble_and_writes_metadata(
    monkeypatch, tmp_path, source_pickle, caplog
):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    for name in ("one.ckpt", "two.ckpt"):
        (checkpoint_dir / name).write_bytes(name.encode())

    models = iter(
        [
            FakeModel(["a", "b"], 0.2, 0.1),
            FakeModel(["a", "b"], 0.6, 0.5),
        ]
    )
    monkeypatch.setattr(
        _teacher_targets,
        "get_config",
        lambda _: SimpleNamespace(
            audio=SimpleNamespace(spec_height=2, spec_width=3),
            infer=SimpleNamespace(scaling_coefficient=99, scaling_intercept=99),
        ),
    )
    monkeypatch.setattr(
        "britekit.models.model_loader.load_from_checkpoint",
        lambda _, **kwargs: next(models),
    )
    monkeypatch.setattr(
        _teacher_targets.util, "expand_spectrogram", lambda value, **kwargs: value
    )

    output_path = tmp_path / "targets.pkl"
    with caplog.at_level("INFO"):
        _teacher_targets.teacher_targets(
            str(source_pickle),
            str(checkpoint_dir),
            str(output_path),
            batch_size=2,
            device="cpu",
        )

    with output_path.open("rb") as file:
        output = pickle.load(file)

    assert output["format_version"] == 2
    assert output["class_codes"] == ["a", "b"]
    assert output["segment_ids"] == [10, 20, 30]
    np.testing.assert_allclose(output["probabilities"], 0.4)
    assert output["frame_probabilities"].shape == (3, 2, 4)
    assert output["frame_probabilities"].dtype == np.float16
    np.testing.assert_allclose(output["frame_probabilities"], 0.3, atol=0.001)
    assert [item["name"] for item in output["teacher"]["checkpoints"]] == [
        "one.ckpt",
        "two.ckpt",
    ]
    assert len(output["source"]["sha256"]) == 64
    assert "Teacher inference: 3/3 spectrograms (100.0%)" in caplog.text
    assert "segment shape (3, 2) and frame shape (3, 2, 4)" in caplog.text


def test_teacher_targets_rejects_class_mismatch(monkeypatch, tmp_path, source_pickle):
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"teacher")
    monkeypatch.setattr(
        _teacher_targets,
        "get_config",
        lambda _: SimpleNamespace(
            audio=SimpleNamespace(spec_height=2, spec_width=3),
            infer=SimpleNamespace(scaling_coefficient=1, scaling_intercept=0),
        ),
    )
    monkeypatch.setattr(
        "britekit.models.model_loader.load_from_checkpoint",
        lambda _, **kwargs: FakeModel(["b", "a"], 0.5),
    )

    with pytest.raises(ValueError, match="class codes"):
        _teacher_targets.teacher_targets(
            str(source_pickle), str(checkpoint), str(tmp_path / "targets.pkl")
        )


def test_teacher_targets_requires_stable_segment_ids(tmp_path):
    source = tmp_path / "training.pkl"
    with source.open("wb") as file:
        pickle.dump({"class_codes": ["a"], "spec_values": [b"spec"]}, file)
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"teacher")

    with pytest.raises(ValueError, match="spec_segment_ids"):
        _teacher_targets.teacher_targets(
            str(source), str(checkpoint), str(tmp_path / "targets.pkl")
        )


@pytest.mark.parametrize("power", [1.0, 2.0])
def test_teacher_targets_preprocesses_each_teacher_independently(
    monkeypatch, tmp_path, source_pickle, power
):
    # Stored features have energy levels 0, -20, -40, -80 dB, silence, -60 dB.
    magnitude = np.array([1, 0.1, 0.01, 0.0001, 0, 0.001], dtype=np.float32)
    source_spec = magnitude**power
    with source_pickle.open("wb") as file:
        pickle.dump(
            dict(
                class_codes=["a", "b"],
                spec_values=[source_spec.copy() for _ in range(3)],
                spec_segment_ids=[10, 20, 30],
            ),
            file,
        )
    models = [
        FakeModel(["a", "b"], 0.2, 0.1, power=power, convert_to_db=True),
        FakeModel(
            ["a", "b"],
            0.4,
            0.3,
            power=power,
            convert_to_db=True,
            top_db=40,
            db_power=2,
        ),
        FakeModel(["a", "b"], 0.6, 0.5, power=power),
    ]
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    for i in range(len(models)):
        (checkpoint_dir / f"{i}.ckpt").write_bytes(b"teacher")

    cfg = SimpleNamespace(
        # Deliberately differ from the checkpoint metadata.
        audio=SimpleNamespace(spec_height=99, spec_width=99, top_db=10),
        infer=SimpleNamespace(scaling_coefficient=99, scaling_intercept=99),
    )
    original_cfg = deepcopy(cfg)
    monkeypatch.setattr(_teacher_targets, "get_config", lambda _: cfg)
    model_iterator = iter(models)

    def load_checkpoint(path, apply_training_config=True):
        assert not apply_training_config
        return next(model_iterator)

    monkeypatch.setattr(
        "britekit.models.model_loader.load_from_checkpoint", load_checkpoint
    )
    monkeypatch.setattr(
        _teacher_targets.util, "expand_spectrogram", lambda value, **kwargs: value
    )
    output_path = tmp_path / "targets.pkl"
    _teacher_targets.teacher_targets(
        str(source_pickle),
        str(checkpoint_dir),
        str(output_path),
        batch_size=2,
        device="cpu",
    )

    expected = [
        [1, 0.75, 0.5, 0, 0, 0.25],
        [1, 0.25, 0, 0, 0, 0],
        source_spec,
    ]
    for model, values in zip(models, expected):
        assert [len(batch) for batch in model.inputs] == [2, 1]
        actual = np.concatenate(model.inputs).reshape(3, 6)
        np.testing.assert_allclose(actual, np.tile(values, (3, 1)), atol=1e-6)
        assert model.cfg.infer.scaling_coefficient == 1
        assert model.cfg.infer.scaling_intercept == 0
    assert cfg == original_cfg
    assert len({id(model.cfg) for model in models}) == len(models)
    np.testing.assert_array_equal(source_spec, magnitude**power)
    with output_path.open("rb") as file:
        targets = pickle.load(file)
    np.testing.assert_allclose(targets["probabilities"], 0.4)
    np.testing.assert_allclose(targets["frame_probabilities"], 0.3, atol=0.001)


def test_teacher_targets_does_not_reconvert_legacy_db_features(
    monkeypatch, tmp_path, source_pickle
):
    model = FakeModel(["a", "b"], 0.5, decibels=True)
    cfg = SimpleNamespace(
        audio=SimpleNamespace(convert_to_db=True),
        infer=SimpleNamespace(scaling_coefficient=1, scaling_intercept=0),
    )
    monkeypatch.setattr(_teacher_targets, "get_config", lambda _: cfg)
    monkeypatch.setattr(
        "britekit.models.model_loader.load_from_checkpoint",
        lambda _, **kwargs: model,
    )
    monkeypatch.setattr(
        _teacher_targets.util, "expand_spectrogram", lambda value, **kwargs: value
    )
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"teacher")
    _teacher_targets.teacher_targets(
        str(source_pickle), str(checkpoint), str(tmp_path / "targets.pkl"), device="cpu"
    )
    np.testing.assert_array_equal(
        model.inputs[0].reshape(3, 6), np.tile(np.arange(6), (3, 1))
    )


@pytest.mark.parametrize("bits", [8, 16])
def test_teacher_targets_matches_training_preprocessing_with_real_checkpoint(
    monkeypatch, tmp_path, bits
):
    import lightning.pytorch as pl
    import torch

    from britekit.core import config_loader, util
    from britekit.core.base_config import BaseConfig
    from britekit.core.dataset import SpectrogramDataset
    from britekit.models import model_loader

    cfg = BaseConfig()
    cfg.audio.spec_height = 32
    cfg.audio.spec_width = 64
    cfg.audio.spec_duration = 2
    cfg.audio.convert_to_db = True
    cfg.audio.top_db = 40
    cfg.audio.db_power = 2
    cfg.train.model_type = "effnet.1"
    cfg.train.head_type = "temporal_sed"
    cfg.train.hidden_channels = 8
    monkeypatch.setattr(config_loader, "_base_config", cfg)
    monkeypatch.setattr(model_loader, "get_device", lambda: "cpu")

    model = model_loader.load_new_model(
        ["Alpha", "Beta"], ["a", "b"], ["", ""], ["", ""], 1
    ).eval()
    values = np.geomspace(0.001, 1, 32 * 64).astype(np.float32).reshape(32, 64)
    specs = [util.compress_spectrogram(values, bits=bits)]
    dataset = SpectrogramDataset(specs, [[0]], 2, is_training=False)
    expected = model.predict(dataset[0]["input"].unsqueeze(0), "cpu")

    checkpoint = tmp_path / "teacher.ckpt"
    state = dict(
        epoch=0,
        state_dict=model.state_dict(),
        hyper_parameters=dict(model.hparams),
        **{"pytorch-lightning_version": pl.__version__},
    )
    model.on_save_checkpoint(state)
    torch.save(state, checkpoint)
    source = tmp_path / "training.pkl"
    with source.open("wb") as file:
        pickle.dump(
            dict(class_codes=["a", "b"], spec_values=specs, spec_segment_ids=[10]),
            file,
        )
    # Metadata must supply the shape and conversion, even with default runtime
    # audio settings. Loading the checkpoint must not mutate these defaults.
    runtime_cfg = BaseConfig()
    monkeypatch.setattr(config_loader, "_base_config", runtime_cfg)
    output_path = tmp_path / "targets.pkl"
    _teacher_targets.teacher_targets(
        str(source), str(checkpoint), str(output_path), device="cpu"
    )
    with output_path.open("rb") as file:
        targets = pickle.load(file)
    np.testing.assert_allclose(targets["probabilities"], expected[0], atol=1e-6)
    np.testing.assert_allclose(targets["frame_probabilities"], expected[1], atol=5e-4)
    assert runtime_cfg == BaseConfig()
