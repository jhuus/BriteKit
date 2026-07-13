#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from britekit.commands import _ensemble


def make_evaluation_tester():
    tester = MagicMock()
    tester.y_pred_trained_df = pd.DataFrame({"": ["rec-0", "rec-1"]})
    # Matches the Fortran-contiguous layout produced by BaseTester.convert_to_numpy().
    tester.y_pred_trained = np.asfortranarray(np.zeros((2, 2), dtype=np.float32))
    tester.segment_len = 3.0
    tester.overlap = 0.0
    tester.trained_classes = ["A", "B"]
    tester.trained_class_indexes = {"A": 0, "B": 1}
    tester.get_auc_metric.return_value = 0.42
    return tester


def test_eval_scores_maps_long_rows_and_calls_only_selected_metric():
    tester = make_evaluation_tester()
    metadata = pd.DataFrame(
        {
            "recording": ["rec", "rec", "rec", "rec"],
            "name": ["A", "A", "B", "B"],
            "start_time": [0.0, 3.0, 0.0, 3.0],
            "end_time": [3.0, 6.0, 3.0, 6.0],
        }
    )
    source_indexes, flat_indexes = _ensemble._build_prediction_mapping(tester, metadata)

    result = _ensemble._eval_scores(
        np.array([0.1, 0.2, 0.3, 0.4]),
        tester,
        "micro_pr",
        source_indexes,
        flat_indexes,
    )

    assert result == 0.42
    metric, predictions = tester.get_auc_metric.call_args.args
    assert metric == "micro_pr"
    np.testing.assert_allclose(predictions, [[0.1, 0.3], [0.2, 0.4]])


def configure_fake_inference(monkeypatch: pytest.MonkeyPatch, dataframes):
    analyzer = MagicMock()
    frames = iter(dataframes)

    def run(_, output_path, rtype):
        assert rtype == "csv"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        next(frames).to_csv(Path(output_path) / "scores.csv", index=False)

    analyzer.return_value.run.side_effect = run
    monkeypatch.setattr("britekit.core.analyzer.Analyzer", analyzer)
    return analyzer


def make_prediction_dataframe(score=0.0, name="A"):
    return pd.DataFrame(
        {
            "recording": ["rec"],
            "name": [name],
            "start_time": [0.0],
            "end_time": [3.0],
            "score": [score],
        }
    )


def test_ensemble_rejects_mismatched_prediction_rows_and_restores_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "one.ckpt").touch()
    (ckpt_dir / "two.ckpt").touch()
    cfg = SimpleNamespace(
        misc=SimpleNamespace(ckpt_folder="original"),
        infer=SimpleNamespace(min_score=0.5),
    )
    monkeypatch.setattr(_ensemble, "get_config", MagicMock(return_value=cfg))
    monkeypatch.setattr(_ensemble.util, "set_logging", MagicMock())
    configure_fake_inference(
        monkeypatch,
        [make_prediction_dataframe(name="A"), make_prediction_dataframe(name="B")],
    )

    with pytest.raises(ValueError, match="do not match"):
        _ensemble.ensemble(
            ckpt_dir=str(ckpt_dir),
            ensemble_size=1,
            annotations_path=str(tmp_path / "annotations.csv"),
            recordings_path=str(tmp_path),
        )

    assert cfg.misc.ckpt_folder == "original"
    assert cfg.infer.min_score == 0.5
    assert not (tmp_path / "ensemble_evaluation_labels").exists()


def test_ensemble_selects_zero_score_and_initializes_tester_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "one.ckpt").touch()
    (ckpt_dir / "two.ckpt").touch()
    save_dir = tmp_path / "saved"
    cfg = SimpleNamespace(
        misc=SimpleNamespace(ckpt_folder="original"),
        infer=SimpleNamespace(min_score=0.5),
    )
    monkeypatch.setattr(_ensemble, "get_config", MagicMock(return_value=cfg))
    monkeypatch.setattr(_ensemble.util, "set_logging", MagicMock())
    configure_fake_inference(
        monkeypatch,
        [make_prediction_dataframe(), make_prediction_dataframe()],
    )

    tester = make_evaluation_tester()
    tester.y_pred_trained_df = pd.DataFrame({"": ["rec-0"]})
    tester.y_pred_trained = np.zeros((1, 1), dtype=np.float32)
    tester.trained_classes = ["A"]
    tester.trained_class_indexes = {"A": 0}
    tester.get_auc_metric.return_value = 0.0
    tester_class = MagicMock(return_value=tester)
    monkeypatch.setattr(
        "britekit.testing.per_segment_tester.PerSegmentTester", tester_class
    )

    _ensemble.ensemble(
        ckpt_dir=str(ckpt_dir),
        ensemble_size=1,
        annotations_path=str(tmp_path / "annotations.csv"),
        recordings_path=str(tmp_path),
        save_dir=str(save_dir),
    )

    assert len(list(save_dir.glob("*.ckpt"))) == 1
    tester_class.assert_called_once()
    assert tester_class.call_args.kwargs["save_matrices"] is False
    assert cfg.misc.ckpt_folder == "original"
    assert cfg.infer.min_score == 0.5
