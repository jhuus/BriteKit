import random

import pytest

from britekit.testing.iou_tester import IoUTester


def _all_pairs_iou_scores(tester, annotations, labels):
    """Reference implementation of the former all-pairs overlap graph."""
    ann_neighbors = [[] for _ in annotations]
    lbl_neighbors = [[] for _ in labels]
    for i, annotation in enumerate(annotations):
        for j, label in enumerate(labels):
            if tester._overlaps(annotation, label):
                ann_neighbors[i].append(j)
                lbl_neighbors[j].append(i)

    ann_visited = [False] * len(annotations)
    lbl_visited = [False] * len(labels)
    scores = []

    for start in range(len(annotations)):
        if ann_visited[start]:
            continue
        component_annotations = []
        component_labels = []
        stack = [(True, start)]
        while stack:
            is_annotation, index = stack.pop()
            if is_annotation:
                if ann_visited[index]:
                    continue
                ann_visited[index] = True
                component_annotations.append(index)
                stack.extend((False, j) for j in ann_neighbors[index])
            else:
                if lbl_visited[index]:
                    continue
                lbl_visited[index] = True
                component_labels.append(index)
                stack.extend((True, i) for i in lbl_neighbors[index])

        if not component_labels:
            scores.append(0.0)
            continue
        merged_annotations = tester._merge(
            [annotations[i] for i in component_annotations]
        )
        merged_labels = tester._merge(
            [(labels[j][0], labels[j][1]) for j in component_labels]
        )
        intersection = tester._intersection_length(merged_annotations, merged_labels)
        union = (
            tester._total_length(merged_annotations)
            + tester._total_length(merged_labels)
            - intersection
        )
        scores.append(intersection / union if union else 0.0)

    scores.extend(0.0 for visited in lbl_visited if not visited)
    return scores


def test_sweep_line_matches_all_pairs_overlap_graph():
    tester = IoUTester.__new__(IoUTester)
    random.seed(42)

    for _ in range(100):
        annotations = sorted(
            (start := random.random() * 20, start + random.random() * 4)
            for _ in range(random.randrange(8))
        )
        labels = sorted(
            (
                start := random.random() * 20,
                start + random.random() * 4,
                random.random(),
            )
            for _ in range(random.randrange(12))
        )

        expected = _all_pairs_iou_scores(tester, annotations, labels)
        actual = tester._iou_scores_for_pair(annotations, labels)

        assert sorted(actual) == pytest.approx(sorted(expected))


def test_iou_details_returns_matching_overall_and_recording_scores():
    tester = IoUTester.__new__(IoUTester)
    annotations = {
        "recording-1": {"A": [(0.0, 2.0)]},
        "recording-2": {"A": [(0.0, 1.0)], "B": [(2.0, 3.0)]},
    }
    labels = {
        "recording-1": {"A": [(0.0, 1.0, 0.9)]},
        "recording-2": {"A": [(0.0, 1.0, 0.9)]},
    }

    overall, recordings = tester._compute_iou_details(annotations, labels, 0.5)

    assert overall == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert recordings == pytest.approx({"recording-1": 0.5, "recording-2": 0.5})


def test_load_labels_only_keeps_annotated_recording_class_pairs(tmp_path):
    (tmp_path / "recording-1_scores.txt").write_text("0\t1\tA;0.9\n0\t1\tB;0.8\n")
    (tmp_path / "recording-2_scores.txt").write_text("0\t1\tA;0.7\n")
    tester = IoUTester.__new__(IoUTester)
    tester.label_dir = str(tmp_path)
    annotations = {"recording-1": {"A": [(0.0, 1.0)]}}

    labels = tester._load_labels(annotations)

    assert labels == {"recording-1": {"A": [(0.0, 1.0, 0.9)]}}
