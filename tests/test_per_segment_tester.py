#!/usr/bin/env python3

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from britekit.commands import _reports
from britekit.testing.base_tester import BaseTester, Label
from britekit.testing.per_segment_tester import Annotation, PerSegmentTester


def make_tester() -> PerSegmentTester:
    tester = PerSegmentTester.__new__(PerSegmentTester)
    BaseTester.__init__(tester)
    tester.segment_len = 3.0
    tester.overlap = 0.0
    tester.trained_classes = ["A", "B", "C"]
    tester.trained_class_set = set(tester.trained_classes)
    tester.annotated_classes = ["A", "C"]
    tester.annotated_class_set = set(tester.annotated_classes)
    tester.set_class_indexes()
    return tester


def test_init_y_true_builds_expected_numpy_backed_dataframes():
    tester = make_tester()
    tester.recording_duration = {"rec": 9.0}
    tester.annotations = {"rec": [Annotation(0.5, 1.0, "A"), Annotation(3.2, 3.8, "C")]}

    tester.init_y_true()

    expected = pd.DataFrame(
        {
            "": ["rec-0", "rec-1", "rec-2"],
            "A": np.array([1, 0, 0], dtype=np.uint8),
            "B": np.array([0, 0, 0], dtype=np.uint8),
            "C": np.array([0, 1, 0], dtype=np.uint8),
        }
    )
    pd.testing.assert_frame_equal(tester.y_true_trained_df, expected)
    assert tester.y_true_annotated_df.columns.tolist() == ["", "A", "C"]
    assert tester.segments_per_recording == {"rec": [0, 1, 2]}


def test_init_y_pred_populates_scores_and_preserves_duplicate_behavior():
    tester = make_tester()
    tester.segments_per_recording = {"rec": [0, 1]}
    first = Label("rec", "A", 0.0, 3.0, 0.8)
    second = Label("rec", "A", 0.0, 3.0, 0.3)
    third = Label("rec", "C", 3.0, 6.0, 0.7)
    first.segment = second.segment = 0
    third.segment = 1
    tester.labels_per_recording = {"rec": [first, second, third]}

    tester.init_y_pred(tester.segments_per_recording, use_max_score=False)
    assert tester.y_pred_trained_df["A"].tolist() == pytest.approx([0.3, 0.0])
    assert tester.y_pred_trained_df["C"].tolist() == pytest.approx([0.0, 0.7])
    assert tester.y_pred_annotated_df.columns.tolist() == ["", "A", "C"]

    tester.init_y_pred(tester.segments_per_recording, use_max_score=True)
    assert tester.y_pred_trained_df["A"].tolist() == pytest.approx([0.8, 0.0])


class NoIterationList(list):
    def __iter__(self):
        raise AssertionError("full recording label list was scanned")


def test_precision_in_seconds_uses_segment_label_index():
    tester = make_tester()
    tester.segments_per_recording = {"rec": [0, 1]}
    label = Label("rec", "A", 0.0, 3.0, 0.9)
    label.segment = 0
    tester.labels_per_recording = {"rec": NoIterationList([label])}
    tester.labels_by_segment = {"rec": {0: [label], 1: []}}
    tester.y_true_annotated = np.array([[1, 0], [0, 0]], dtype=np.float32)

    precision, tp_secs, fp_secs, _, _ = tester._calc_precision_in_seconds(0.5)

    assert precision == 1.0
    assert tp_secs == 3.0
    assert fp_secs == 0.0


@pytest.mark.parametrize(
    ("metric", "stats_method", "stats_key"),
    [
        ("macro_pr", "get_pr_auc_stats", "macro_pr_auc"),
        ("micro_pr", "get_pr_auc_stats", "micro_pr_auc_trained"),
        ("macro_roc", "get_roc_auc_stats", "macro_roc_auc"),
        ("micro_roc", "get_roc_auc_stats", "micro_roc_auc_trained"),
    ],
)
def test_get_auc_metric_matches_full_statistics(
    metric: str, stats_method: str, stats_key: str
):
    tester = make_tester()
    tester.y_true_trained = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float32
    )
    tester.y_pred_trained = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.3, 0.8, 0.2],
            [0.2, 0.1, 0.7],
            [0.8, 0.3, 0.6],
        ],
        dtype=np.float32,
    )
    tester.y_true_annotated = tester.y_true_trained[:, [0, 2]]
    tester.y_pred_annotated = tester.y_pred_trained[:, [0, 2]]

    expected = getattr(tester, stats_method)()[stats_key]

    assert tester.get_auc_metric(metric) == pytest.approx(expected)


@pytest.mark.parametrize("save_matrices", [True, False])
def test_initialize_can_skip_matrix_csvs(
    tmp_path: Path, save_matrices: bool, monkeypatch: pytest.MonkeyPatch
):
    tester = make_tester()
    tester.output_dir = str(tmp_path)
    tester.label_dir = "unused"
    tester.recording_dir = "unused"
    tester.save_matrices = save_matrices
    tester.segments_per_recording = {"rec": [0]}
    monkeypatch.setattr(tester, "get_labels", MagicMock())
    monkeypatch.setattr(tester, "get_recording_info", MagicMock())
    monkeypatch.setattr(tester, "get_annotations", MagicMock())
    monkeypatch.setattr(tester, "init_y_true", MagicMock())
    monkeypatch.setattr(tester, "init_y_pred", MagicMock())
    monkeypatch.setattr(tester, "convert_to_numpy", MagicMock())
    monkeypatch.setattr(tester, "check_if_arrays_match", MagicMock())
    matrix = pd.DataFrame({"": ["rec-0"], "A": [1]})
    tester.y_true_annotated_df = matrix
    tester.y_pred_annotated_df = matrix
    tester.y_true_trained_df = matrix
    tester.y_pred_trained_df = matrix

    tester.initialize()

    expected_files = {
        "y_true_annotated.csv",
        "y_pred_annotated.csv",
        "y_true_trained.csv",
        "y_pred_trained.csv",
    }
    written_files = {path.name for path in tmp_path.iterdir()}
    assert written_files == (expected_files if save_matrices else set())


def test_rpt_test_forwards_skip_matrices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_path = tmp_path / "labels"
    labels_path.mkdir()
    tester_class = MagicMock()
    cfg = MagicMock()
    cfg.infer.min_score = 0.2
    monkeypatch.setattr(_reports, "get_config", MagicMock(return_value=cfg))
    monkeypatch.setattr(
        "britekit.testing.per_segment_tester.PerSegmentTester", tester_class
    )

    _reports.rpt_test(
        granularity="segment",
        annotations_path=str(tmp_path / "annotations.csv"),
        label_dir="labels",
        output_path=str(tmp_path / "output"),
        recordings_path=str(tmp_path),
        save_matrices=False,
    )

    tester_class.assert_called_once_with(
        str(tmp_path / "annotations.csv"),
        str(tmp_path),
        str(labels_path),
        str(tmp_path / "output"),
        0.2,
        save_matrices=False,
    )
    tester_class.return_value.run.assert_called_once_with()
