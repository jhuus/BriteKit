import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from britekit.commands import _teacher_targets


class FakeModel:
    def __init__(self, class_codes, score, frame_score=None):
        self.train_class_codes = class_codes
        self.score = score
        self.frame_score = frame_score

    def eval(self):
        return self

    def set_config(self, cfg):
        self.cfg = cfg

    def to(self, device):
        return self

    def predict(self, specs, device):
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
        "britekit.models.model_loader.load_from_checkpoint", lambda _: next(models)
    )
    monkeypatch.setattr(
        _teacher_targets.util, "expand_spectrogram", lambda value: value
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
        lambda _: FakeModel(["b", "a"], 0.5),
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
