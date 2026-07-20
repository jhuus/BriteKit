#!/usr/bin/env python3

import logging
import os
from typing import Optional

import numpy as np

from britekit.core.config_loader import get_config
from britekit.core import util


class IoUTester:
    """
    Measure temporal localization quality using Intersection-over-Union (IoU).

    Annotations may be incomplete: for a given recording, if a class is annotated,
    all occurrences of that class in that recording are annotated, but other classes
    may be present without annotation. Only annotated (recording, class) pairs are scored.

    At each threshold, labels with score >= threshold are compared against annotations.
    Overlapping annotations and labels are grouped into connected components; each
    component yields one IoU score (intersection / union). Isolated annotations and
    isolated labels each yield a score of 0. The total IoU is the mean of all scores.
    """

    def __init__(
        self,
        annotation_path: str,
        label_dir: str,
        output_dir: str,
        start: float = 0.0,
        end: float = 1.0,
        incr: float = 0.01,
        t1: float = 0.6,
        t2: float = 0.8,
        cfg_path: Optional[str] = None,
    ):
        self.annotation_path = annotation_path
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.start = start
        self.end = end
        self.incr = incr
        self.t1 = t1
        self.t2 = t2
        self.cfg = get_config(cfg_path)

    def _load_annotations(self):
        """Return {recording: {class_code: [(start, end)]}} from the annotations CSV."""
        import pandas as pd

        df = pd.read_csv(
            self.annotation_path,
            dtype={"recording": str, "class": str},
            keep_default_na=False,
        )
        annotations: dict = {}
        for _, row in df.iterrows():
            recording = row["recording"]
            if not recording:
                continue

            class_code = row["class"]
            if not class_code:
                continue

            if self.cfg.misc.map_codes and class_code in self.cfg.misc.map_codes:
                class_code = self.cfg.misc.map_codes[class_code]

            if recording not in annotations:
                annotations[recording] = {}
            if class_code not in annotations[recording]:
                annotations[recording][class_code] = []

            annotations[recording][class_code].append(
                (float(row["start_time"]), float(row["end_time"]))
            )

        for class_dict in annotations.values():
            for intervals in class_dict.values():
                intervals.sort(key=lambda interval: interval[0])

        return annotations

    def _load_labels(self, annotations=None):
        """Return {recording: {class_code: [(start, end, score)]}} from label files."""
        recording_classes = None
        if annotations is not None:
            recording_classes = {
                recording: set(class_dict)
                for recording, class_dict in annotations.items()
            }
        label_df = util.inference_output_to_dataframe(
            self.label_dir, recording_classes=recording_classes
        )
        labels: dict = {}
        for _, row in label_df.iterrows():
            recording = row["recording"]
            class_code = row["name"]

            if recording not in labels:
                labels[recording] = {}
            if class_code not in labels[recording]:
                labels[recording][class_code] = []

            labels[recording][class_code].append(
                (float(row["start_time"]), float(row["end_time"]), float(row["score"]))
            )

        for class_dict in labels.values():
            for intervals in class_dict.values():
                intervals.sort(key=lambda interval: interval[0])

        return labels

    @staticmethod
    def _merge_adjacent_labels(lbl_intervals):
        """
        Merge contiguous or overlapping label intervals, keeping the maximum score.
        Contiguous means end of one label == start of the next (as produced by fixed-
        width analysis frames). The merged interval spans the full run; its score is
        the maximum of the individual scores.
        """
        if not lbl_intervals:
            return []
        # Label intervals are sorted once when loaded. Filtering by threshold
        # preserves that order, avoiding a sort for every threshold.
        merged = [list(lbl_intervals[0])]
        for s, e, score in lbl_intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
                merged[-1][2] = max(merged[-1][2], score)
            else:
                merged.append([s, e, score])
        return [(s, e, sc) for s, e, sc in merged]

    @staticmethod
    def _overlaps(a, b):
        """True if intervals a and b share any duration."""
        return a[0] < b[1] and b[0] < a[1]

    @staticmethod
    def _merge(intervals):
        """Merge a list of (start, end) pairs into sorted, non-overlapping intervals."""
        if not intervals:
            return []
        ivs = sorted(intervals, key=lambda x: x[0])
        merged = [[ivs[0][0], ivs[0][1]]]
        for s, e in ivs[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    @staticmethod
    def _total_length(merged):
        return sum(e - s for s, e in merged)

    @staticmethod
    def _intersection_length(merged_a, merged_b):
        """Length of the intersection of two sorted, non-overlapping interval lists."""
        total = 0.0
        i, j = 0, 0
        while i < len(merged_a) and j < len(merged_b):
            lo = max(merged_a[i][0], merged_b[j][0])
            hi = min(merged_a[i][1], merged_b[j][1])
            if lo < hi:
                total += hi - lo
            if merged_a[i][1] < merged_b[j][1]:
                i += 1
            else:
                j += 1
        return total

    def _iou_scores_for_pair(self, ann_intervals, lbl_intervals):
        """
        Compute IoU scores for one (recording, class) pair.

        Builds a bipartite graph where annotations and labels are nodes and edges
        connect overlapping pairs. Each connected component produces one score:
        the IoU of all annotation intervals in the component vs. all label intervals.
        Isolated annotations and isolated labels each produce a score of 0.
        """
        ann_neighbors = [[] for _ in ann_intervals]
        lbl_neighbors = [[] for _ in lbl_intervals]
        # Both lists are sorted by start time. Advance past labels that end
        # before each annotation, then inspect only labels that could overlap.
        # This replaces the previous all-pairs O(A * L) scan.
        first_possible_label = 0
        for i, ann in enumerate(ann_intervals):
            while (
                first_possible_label < len(lbl_intervals)
                and lbl_intervals[first_possible_label][1] <= ann[0]
            ):
                first_possible_label += 1

            j = first_possible_label
            while j < len(lbl_intervals) and lbl_intervals[j][0] < ann[1]:
                if self._overlaps(ann, lbl_intervals[j]):
                    ann_neighbors[i].append(j)
                    lbl_neighbors[j].append(i)
                j += 1

        ann_visited = [False] * len(ann_intervals)
        lbl_visited = [False] * len(lbl_intervals)
        scores = []

        def traverse(start_kind, start_idx):
            comp_anns, comp_lbls = [], []
            stack = [(start_kind, start_idx)]
            while stack:
                kind, idx = stack.pop()
                if kind == "a":
                    if ann_visited[idx]:
                        continue
                    ann_visited[idx] = True
                    comp_anns.append(idx)
                    for j in ann_neighbors[idx]:
                        if not lbl_visited[j]:
                            stack.append(("l", j))
                else:
                    if lbl_visited[idx]:
                        continue
                    lbl_visited[idx] = True
                    comp_lbls.append(idx)
                    for i in lbl_neighbors[idx]:
                        if not ann_visited[i]:
                            stack.append(("a", i))
            return comp_anns, comp_lbls

        for i in range(len(ann_intervals)):
            if ann_visited[i]:
                continue
            comp_anns, comp_lbls = traverse("a", i)
            if not comp_lbls:
                scores.append(0.0)
            else:
                merged_ann = self._merge([ann_intervals[k] for k in comp_anns])
                merged_lbl = self._merge(
                    [(lbl_intervals[k][0], lbl_intervals[k][1]) for k in comp_lbls]
                )
                inter = self._intersection_length(merged_ann, merged_lbl)
                union = (
                    self._total_length(merged_ann)
                    + self._total_length(merged_lbl)
                    - inter
                )
                scores.append(inter / union if union > 0 else 0.0)

        for j in range(len(lbl_intervals)):
            if not lbl_visited[j]:
                scores.append(0.0)

        return scores

    def _compute_iou_details(self, annotations, labels, threshold):
        """Return overall and per-recording IoU at the given threshold."""
        all_scores = []
        scores_by_recording = {}
        debug = logging.getLogger().isEnabledFor(logging.DEBUG)
        if debug:
            logging.debug("threshold=%.4f", threshold)

        for recording, class_dict in annotations.items():
            recording_scores = []
            for class_code, ann_intervals in class_dict.items():
                raw_lbls = []
                lbl_intervals = []
                if recording in labels and class_code in labels[recording]:
                    raw_lbls = [
                        iv for iv in labels[recording][class_code] if iv[2] >= threshold
                    ]
                    lbl_intervals = self._merge_adjacent_labels(raw_lbls)

                if debug:
                    logging.debug(
                        f"  {recording} / {class_code}:"
                        f"\n    annotations : {ann_intervals}"
                        f"\n    labels (raw): {raw_lbls}"
                        f"\n    labels (merged): {[(s, e) for s, e, _ in lbl_intervals]}"
                    )

                scores = self._iou_scores_for_pair(ann_intervals, lbl_intervals)
                if debug:
                    logging.debug(
                        f"    component scores: {[f'{s:.4f}' for s in scores]}"
                    )
                all_scores.extend(scores)
                recording_scores.extend(scores)

            scores_by_recording[recording] = (
                sum(recording_scores) / len(recording_scores)
                if recording_scores
                else 0.0
            )

        if not all_scores:
            if debug:
                logging.debug("  no scores — returning 0.0")
            return 0.0, scores_by_recording

        mean_iou = sum(all_scores) / len(all_scores)
        if debug:
            logging.debug(
                f"  all scores: {[f'{s:.4f}' for s in all_scores]}"
                f"\n  mean IoU: {mean_iou:.4f}"
            )
        return mean_iou, scores_by_recording

    def _compute_iou(self, annotations, labels, threshold):
        """Mean IoU across all annotated (recording, class) pairs at the given threshold."""
        return self._compute_iou_details(annotations, labels, threshold)[0]

    def _compute_iou_per_recording(self, annotations, labels, threshold):
        """Mean IoU per recording across all annotated classes at the given threshold."""
        return self._compute_iou_details(annotations, labels, threshold)[1]

    def _average_tp_score(self, annotations, labels):
        """Average score of raw label intervals that overlap at least one annotation."""
        tp_scores = []
        for recording, class_dict in annotations.items():
            for class_code, ann_intervals in class_dict.items():
                if recording not in labels or class_code not in labels[recording]:
                    continue
                for lbl in labels[recording][class_code]:
                    lbl_iv = (lbl[0], lbl[1])
                    if any(self._overlaps(ann, lbl_iv) for ann in ann_intervals):
                        tp_scores.append(lbl[2])
        return sum(tp_scores) / len(tp_scores) if tp_scores else 0.0

    def run(self):
        import pandas as pd

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        logging.info("Loading annotations")
        annotations = self._load_annotations()

        logging.info("Loading labels")
        labels = self._load_labels(annotations)

        # Build thresholds without floating-point drift
        n = round((self.end - self.start) / self.incr) + 1
        thresholds = [round(self.start + i * self.incr, 6) for i in range(n)]

        results = []
        per_recording_by_threshold = {}
        for threshold in thresholds:
            iou, per_recording = self._compute_iou_details(
                annotations, labels, threshold
            )
            results.append((threshold, iou))
            per_recording_by_threshold[threshold] = per_recording

        max_iou = max(r[1] for r in results) if results else 0.0
        max_threshold = next(r[0] for r in results if r[1] == max_iou)

        results_by_threshold = dict(results)

        def get_threshold_details(threshold):
            if threshold in results_by_threshold:
                return (
                    results_by_threshold[threshold],
                    per_recording_by_threshold[threshold],
                )
            return self._compute_iou_details(annotations, labels, threshold)

        iou_t1, per_t1 = get_threshold_details(self.t1)
        iou_t2, per_t2 = get_threshold_details(self.t2)

        thresholds_arr = np.array([r[0] for r in results])
        iou_arr = np.array([r[1] for r in results])
        auc = float(np.trapezoid(iou_arr, thresholds_arr)) if len(results) > 1 else 0.0
        avg_tp_score = self._average_tp_score(annotations, labels)

        pd.DataFrame(results, columns=["threshold", "iou"]).to_csv(
            os.path.join(self.output_dir, "thresholds.csv"),
            index=False,
            float_format="%.4f",
        )

        per_peak = per_recording_by_threshold[max_threshold]
        recordings_df = pd.DataFrame(
            {
                "recording": sorted(per_peak.keys()),
                "iou_max": [per_peak[r] for r in sorted(per_peak.keys())],
                f"iou_{self.t1:.2f}": [per_t1[r] for r in sorted(per_peak.keys())],
                f"iou_{self.t2:.2f}": [per_t2[r] for r in sorted(per_peak.keys())],
            }
        )
        recordings_df.to_csv(
            os.path.join(self.output_dir, "recordings.csv"),
            index=False,
            float_format="%.4f",
        )

        summary_lines = [
            f"IoU at t1 ({self.t1:.4f}) = {iou_t1:.4f}",
            f"IoU at t2 ({self.t2:.4f}) = {iou_t2:.4f}",
            f"Maximum IoU = {max_iou:.4f} at threshold = {max_threshold:.4f}",
            f"IoU AUC = {auc:.4f}",
            f"Average TP score = {avg_tp_score:.4f}",
        ]
        for line in summary_lines:
            logging.info(line)
        with open(os.path.join(self.output_dir, "summary.txt"), "w") as f:
            f.write("\n".join(summary_lines) + "\n")
