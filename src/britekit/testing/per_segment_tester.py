# Defer some imports to improve initialization performance.
import logging
import math
import os
from pathlib import Path
from typing import Optional

from britekit.core.config_loader import get_config
from britekit.core import util
from britekit.testing.base_tester import BaseTester


class Annotation:
    def __init__(self, start_time, end_time, class_code):
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.class_code = class_code

    def __str__(self):
        return f"{self.class_code}: {self.start_time}-{self.end_time}"


class PerSegmentTester(BaseTester):
    """
    Calculate test metrics when individual sounds are annotated in the ground truth data.
    Annotations are read as a CSV with four columns: recording, class, start_time and end_time.
    The recording column is the file name without the path or type suffix, e.g. "recording1".
    The class column contains the class code, and start_time and end_time are
    fractional seconds, e.g. 12.5 represents 12.5 seconds from the start of the recording.
    If your annotations are in a different format, simply convert to this format to use this script.

    Classifiers should be run with a threshold of 0, and with label merging disabled so segment-specific scores are retained.

    Attributes:
        annotation_path (str): Annotations CSV file.
        recording_dir (str): Directory containing recordings.
        label_dir (str): Directory containing Audacity labels.
        output_dir (str): Output directory, where reports will be written.
        threshold (float): Score threshold for precision/recall reporting.
        trained_classes (list): List of trained class codes.
    """

    def __init__(
        self,
        annotation_path: str,
        recording_dir: str,
        label_dir: str,
        output_dir: str,
        threshold: float,
        calibrate: bool = False,
        cutoff: float = 0.6,
        coef: Optional[float] = None,
        inter: Optional[float] = None,
    ):
        """
        Initialize the PerSegmentTester.

        See class docstring for detailed parameter descriptions and usage information.
        """
        super().__init__()
        self.annotation_path = annotation_path
        self.recording_dir = recording_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.calibrate = calibrate
        self.calibration_cutoff = cutoff
        self.coefficient = coef
        self.intercept = inter

        self.cfg = get_config()

    def _get_recording_info(self):
        """
        Create a dict with the duration in seconds of every recording
        """
        import librosa

        self.recording_duration = {}
        self.recordings = []
        recordings = sorted(util.get_audio_files(self.recording_dir))
        for recording in recordings:
            duration = librosa.get_duration(path=recording)
            self.recording_duration[Path(recording).stem] = duration
            self.recordings.append(Path(recording).stem)

        self.recordings = sorted(self.recordings)

    @staticmethod
    def get_offsets(start_time, end_time, segment_len, overlap, min_seconds=0.3):
        """
        Determine which offsets an annotation or label should be assigned to.

        This static method calculates the time offsets where an annotation or label should
        be assigned based on the segment boundaries. The returned offsets are aligned on
        boundaries of segment_len - overlap.

        Args:
            start_time: Start time of the annotation in seconds
            end_time: End time of the annotation in seconds
            segment_len: Length of each segment in seconds
            overlap: Overlap between consecutive segments in seconds
            min_seconds: Minimum number of seconds that must be contained in the first
                        and last segments (default: 0.3)

        Returns:
            list: List of time offsets where the annotation should be assigned

        Note:
            For example, if segment_len=3 and overlap=1.5, segments are aligned on
            1.5 second boundaries (0, 1.5, 3.0, ...). The method ensures that the
            first and last segments contain at least min_seconds of the labelled sound.
        """

        step = segment_len - overlap
        if step <= 0:
            raise ValueError(
                "segment_len must be greater than overlap to ensure positive step size"
            )

        # find the first aligned offset no more than (segment_len - min_seconds) before start_time,
        # to ensure the first segment contains at least min_seconds of the labelled sound
        first_offset = max(
            0, math.ceil((start_time - (segment_len - min_seconds)) / step) * step
        )

        # generate the list of offsets
        offsets = []
        current_offset = first_offset
        while end_time - current_offset >= min_seconds:
            offsets.append(current_offset)
            current_offset += step

        return offsets

    def get_segments(self, start_time, end_time, min_seconds=0.3):
        """
        Convert time offsets to segment indexes.

        This method converts the time offsets returned by get_offsets() into segment
        indexes that correspond to the actual segments in the analysis.

        Args:
            start_time: Start time of the annotation in seconds
            end_time: End time of the annotation in seconds
            min_seconds: Minimum number of seconds that must be contained in segments
                        (default: 0.3)

        Returns:
            list: List of segment indexes where the annotation should be assigned

        Note:
            Uses self.segment_len and self.overlap to calculate segment boundaries.
            Returns an empty list if no valid segments are found.
        """

        offsets = self.get_offsets(
            start_time, end_time, self.segment_len, self.overlap, min_seconds
        )
        if len(offsets) > 0:
            first_segment = int(offsets[0] // (self.segment_len - self.overlap))
            return [i for i in range(first_segment, first_segment + len(offsets), 1)]
        else:
            return []

    def get_annotations(self):
        """
        Load annotation data from CSV file and process into internal format.

        This method reads a CSV file containing ground truth annotations where each row
        represents a recording, class, start time, and end time. The CSV should have columns:
        "recording" (filename without path/extension), "class" (class code),
        "start_time" (fractional seconds from start), and "end_time" (fractional seconds from start).

        The method processes the annotations, handles class code mapping, filters out
        unknown classes, and organizes the data into Annotation objects for subsequent analysis.

        Note:
            Sets self.annotations, self.annotated_class_set, and self.annotated_classes.
            Calls self.set_class_indexes() to update class indexing.
        """
        import pandas as pd

        # read the annotations
        unknown_classes = set()  # so we only report each unknown class once
        self.annotations = {}
        self.annotated_class_set = set()
        df = pd.read_csv(
            self.annotation_path,
            dtype={"recording": str, "class": str},
            keep_default_na=False,
        )
        for i, row in df.iterrows():
            recording = row["recording"]
            if not recording:
                break

            if recording not in self.annotations:
                self.annotations[recording] = []

            class_code = row["class"]
            if not class_code:
                # useful when including a recording with no annotations
                continue

            if self.cfg.misc.map_codes and class_code in self.cfg.misc.map_codes:
                class_code = self.cfg.misc.map_codes[class_code]

            if class_code not in self.trained_class_set:
                if class_code not in unknown_classes:
                    logging.error(
                        f"Unknown class {class_code} will be skipped (is it in ignore list?)."
                    )
                    unknown_classes.add(class_code)  # so we don't report it again

                continue  # exclude from saved annotations

            annotation = Annotation(row["start_time"], row["end_time"], class_code)
            self.annotations[recording].append(annotation)
            self.annotated_class_set.add(annotation.class_code)

        self.annotated_classes = sorted(self.annotated_class_set)
        self.set_class_indexes()

    def _init_y_true(self):
        """
        Create a dataframe representing the ground truth data, with recordings segmented into 3-second segments
        """
        import numpy as np
        import pandas as pd

        # set segment_dict[recording][segment] = {classes in that segment},
        # where each segment is 3 seconds (self.segment_len) long
        self.segments_per_recording = {}
        segment_dict = {}
        for recording in self.annotations:
            if recording not in self.recording_duration:
                logging.error(
                    f"Ignoring recording {recording} (annotated but not found)."
                )
                continue

            # calculate num_segments exactly as it's done in analyze.py so they match
            increment = self.segment_len - self.overlap
            offsets = np.arange(
                0,
                self.recording_duration[recording] - self.segment_len + 1.0,
                increment,
            ).tolist()

            num_segments = len(offsets)
            self.segments_per_recording[recording] = [i for i in range(num_segments)]
            segment_dict[recording] = {}
            for segment in range(num_segments):
                segment_dict[recording][segment] = {}

            for annotation in self.annotations[recording]:
                segments = self.get_segments(annotation.start_time, annotation.end_time)
                for segment in segments:
                    if segment in segment_dict[recording]:
                        segment_dict[recording][segment][annotation.class_code] = 1

        # convert to 2D array with a row per segment and a column per class;
        # set cells to 1 if class is present and 0 if not present
        self.recordings = []  # base class needs array with recording per row
        rows = []
        for recording in sorted(segment_dict.keys()):
            for segment in sorted(segment_dict[recording].keys()):
                self.recordings.append(recording)
                row = [f"{recording}-{segment}"]
                row.extend([0 for class_code in self.trained_classes])
                for i, class_code in enumerate(self.trained_classes):
                    if class_code in segment_dict[recording][segment]:
                        row[self.trained_class_indexes[class_code] + 1] = 1

                rows.append(row)

        self.y_true_trained_df = pd.DataFrame(rows, columns=[""] + self.trained_classes)

        # create version for annotated classes only
        self.y_true_annotated_df = self.y_true_trained_df.copy()
        for i, column in enumerate(self.y_true_annotated_df.columns):
            if i == 0:
                continue  # skip the index column

            if column not in self.annotated_class_set:
                self.y_true_annotated_df = self.y_true_annotated_df.drop(column, axis=1)

    def _output_pr_per_threshold(self, threshold, precision, recall, name):
        """
        Output precision/recall per threshold
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame()
        df["threshold"] = pd.Series(threshold)
        df["precision"] = pd.Series(precision)
        df["recall"] = pd.Series(recall)
        df.to_csv(
            os.path.join(self.output_dir, f"{name}.csv"),
            index=False,
            float_format="%.3f",
        )

        plt.clf()
        plt.plot(precision, label="Precision")
        plt.plot(recall, label="Recall")
        x_tick_locations, x_tick_labels = [], []
        for i in range(11):
            x_tick_locations.append(int(i * (len(threshold) / 10)))
            x_tick_labels.append(f"{i / 10:.1f}")

        plt.xticks(x_tick_locations, x_tick_labels)
        plt.xlabel("Threshold")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{name}.png"))
        plt.close()

    def _output_pr_curve(self, precision, recall, name):
        """
        Output recall per precision
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame()
        df["precision"] = pd.Series(precision)
        df["recall"] = pd.Series(recall)
        df.to_csv(
            os.path.join(self.output_dir, f"{name}.csv"),
            index=False,
            float_format="%.3f",
        )

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(precision, recall, linewidth=2.0)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        plt.savefig(os.path.join(self.output_dir, f"{name}.png"))
        plt.close()

    def _output_roc_curves(self, threshold, tpr, fpr, precision, recall, suffix):
        """
        Output various ROC curves
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        df = pd.DataFrame()
        df["threshold"] = pd.Series(
            np.flip(threshold)[:-1]
        )  # [:-1] drops the final [inf,0,0] row
        df["true positive rate"] = pd.Series(np.flip(tpr)[:-1])
        df["false positive rate"] = pd.Series(np.flip(fpr)[:-1])
        df.to_csv(
            os.path.join(self.output_dir, f"roc_per_threshold_{suffix}.csv"),
            index=False,
            float_format="%.3f",
        )

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(np.flip(fpr), np.flip(tpr), linewidth=2.0)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        plt.savefig(os.path.join(self.output_dir, f"roc_classic_curve_{suffix}.png"))
        plt.close()

        # flip the axes of the ROC curve so recall is on the x axis and add a precision line
        one_minus_fpr = 1 - fpr
        roc_precision = np.flip(precision)
        if len(recall) > len(tpr):
            roc_precision = roc_precision[: -(len(recall) - len(tpr))]

        df = pd.DataFrame()
        roc_recall, one_minus_fpr = self.interpolate(tpr[:-1], one_minus_fpr[:-1])
        _, roc_precision = self.interpolate(tpr[:-1], roc_precision[:-1])

        # append the straight lines at the end of the curves
        roc_recall_suffix = np.arange(roc_recall[-1] + 0.01, 1.01, 0.01)
        decrement = one_minus_fpr[-1] / len(roc_recall_suffix)
        one_minus_fpr_suffix = np.arange(
            one_minus_fpr[-1] - decrement, -decrement, -decrement
        )
        one_minus_fpr_suffix[-1] = 0
        decrement = roc_precision[-1] / len(roc_recall_suffix)
        roc_precision_suffix = np.arange(
            roc_precision[-1] - decrement, -decrement, -decrement
        )
        roc_precision_suffix[-1] = 0

        roc_recall = np.append(roc_recall, roc_recall_suffix)
        one_minus_fpr = np.append(one_minus_fpr, one_minus_fpr_suffix)
        roc_precision = np.append(roc_precision, roc_precision_suffix)

        df["roc_recall"] = pd.Series(roc_recall)
        df["one_minus_fpr"] = pd.Series(one_minus_fpr)
        df["roc_precision"] = pd.Series(roc_precision)
        df.to_csv(
            os.path.join(self.output_dir, f"roc_inverted_curve_{suffix}.csv"),
            index=False,
            float_format="%.3f",
        )

        plt.clf()
        plt.plot(one_minus_fpr, label="1 - FPR")
        plt.plot(roc_precision, label="Precision")
        x_tick_locations, x_tick_labels = [], []
        for i in range(11):
            x_tick_locations.append(int(i * (len(roc_recall) / 10)))
            x_tick_labels.append(f"{i / 10:.1f}")

        plt.xticks(x_tick_locations, x_tick_labels)
        plt.xlabel("Recall")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"roc_inverted_curve_{suffix}.png"))
        plt.close()

    def _produce_reports(self):
        """
        Produce reports
        """
        import numpy as np
        import pandas as pd

        # calculate and output precision/recall per threshold
        threshold_annotated = self.pr_table_dict["annotated_thresholds"]
        precision_annotated = self.pr_table_dict["annotated_precisions"]
        recall_annotated = self.pr_table_dict["annotated_recalls"]
        self._output_pr_per_threshold(
            threshold_annotated,
            precision_annotated,
            recall_annotated,
            "pr_per_threshold_annotated",
        )

        threshold_trained = self.pr_table_dict["trained_thresholds"]
        precision_trained = self.pr_table_dict["trained_precisions"]
        recall_trained = self.pr_table_dict["trained_recalls"]
        self._output_pr_per_threshold(
            threshold_trained,
            precision_trained,
            recall_trained,
            "pr_per_threshold_trained",
        )

        # calculate and output recall per precision
        self._output_pr_curve(
            precision_annotated, recall_annotated, "pr_curve_annotated"
        )
        self._output_pr_curve(precision_trained, recall_trained, "pr_curve_trained")

        # output the ROC curves
        roc_thresholds = self.roc_auc_dict["roc_thresholds_annotated"]
        roc_tpr = self.roc_auc_dict["roc_tpr_annotated"]
        roc_fpr = self.roc_auc_dict["roc_fpr_annotated"]
        self._output_roc_curves(
            roc_thresholds,
            roc_tpr,
            roc_fpr,
            precision_annotated,
            recall_annotated,
            "annotated",
        )

        roc_thresholds = self.roc_auc_dict["roc_thresholds_trained"]
        roc_tpr = self.roc_auc_dict["roc_tpr_trained"]
        roc_fpr = self.roc_auc_dict["roc_fpr_trained"]
        self._output_roc_curves(
            roc_thresholds,
            roc_tpr,
            roc_fpr,
            precision_trained,
            recall_trained,
            "trained",
        )

        # output a CSV with number of predictions in ranges [0, .1), [.1, .2), ..., [.9, 1.0]
        scores = np.sort(self.prediction_scores)
        prev_idx = 0
        count = []
        for x in np.arange(0.1, 0.91, 0.1):
            idx = np.searchsorted(scores, x)
            count.append(idx - prev_idx)
            prev_idx = idx

        count.append(len(scores) - prev_idx)  # add the count for [.9, 1.0]
        min_value = np.arange(0, 0.91, 0.1)
        max_value = np.arange(0.1, 1.01, 0.1)
        df = pd.DataFrame()
        df["min"] = pd.Series(min_value)
        df["max"] = pd.Series(max_value)
        df["count"] = count
        df.to_csv(
            os.path.join(self.output_dir, "prediction_range_counts.csv"),
            index=False,
            float_format="%.1f",
        )

        # output a summary report
        rpt = []
        rpt.append("For annotated classes only:\n")

        rpt.append(
            f"   Macro-averaged PR-AUC score = {self.pr_auc_dict['macro_pr_auc']:.4f}\n"
        )
        rpt.append(
            f"   Micro-averaged PR-AUC score = {self.pr_auc_dict['micro_pr_auc_annotated']:.4f}\n"
        )
        rpt.append(
            f"   Macro-averaged ROC-AUC score = {self.roc_auc_dict['macro_roc_auc']:.4f}\n"
        )
        rpt.append(
            f"   Micro-averaged ROC-AUC score = {self.roc_auc_dict['micro_roc_auc_annotated']:.4f}\n"
        )
        rpt.append(f"   For threshold = {self.threshold}:\n")
        rpt.append(
            f"      Precision = {100 * self.details_dict['precision_annotated']:.2f}%\n"
        )
        rpt.append(
            f"      Recall = {100 * self.details_dict['recall_annotated']:.2f}%\n"
        )

        rpt.append("\n")
        rpt.append("For all trained classes:\n")
        rpt.append(
            f"   Micro-averaged PR-AUC score = {self.pr_auc_dict['micro_pr_auc_trained']:.4f}\n"
        )
        rpt.append(
            f"   Micro-averaged ROC-AUC score = {self.roc_auc_dict['micro_roc_auc_trained']:.4f}\n"
        )
        rpt.append(f"   For threshold = {self.threshold}:\n")
        rpt.append(
            f"      Precision = {100 * self.details_dict['precision_trained']:.2f}%\n"
        )
        rpt.append(f"      Recall = {100 * self.details_dict['recall_trained']:.2f}%\n")
        rpt.append("\n")
        rpt.append(
            f"Average of macro-PR-AUC-annotated and micro-PR-AUC-trained = {self.pr_auc_dict['combined_pr_auc_trained']:.4f}\n"
        )
        rpt.append(
            f"Average of macro-ROC-annotated and micro-ROC-trained = {self.roc_auc_dict['combined_roc_auc_trained']:.4f}\n"
        )

        logging.info("")
        with open(os.path.join(self.output_dir, "summary_report.txt"), "w") as summary:
            for rpt_line in rpt:
                logging.info(rpt_line[:-1])
                summary.write(rpt_line)

        # write recording details (row per segment)
        recording_summary = []
        rpt_path = os.path.join(self.output_dir, "recording_details.csv")
        with open(rpt_path, "w") as file:
            file.write(
                "recording,segment,TP count,FP count,FN count,TP class,FP class,FN class\n"
            )
            for recording in self.details_dict["rec_info"]:
                total_tp_count = 0
                total_fp_count = 0
                total_fn_count = 0

                for segment in self.segments_per_recording[recording]:
                    tp_seconds = self.details_dict["rec_info"][recording][segment][
                        "tp_seconds"
                    ]
                    tp_count = tp_seconds / self.segment_len
                    total_tp_count += tp_count

                    fp_seconds = self.details_dict["rec_info"][recording][segment][
                        "fp_seconds"
                    ]
                    fp_count = fp_seconds / self.segment_len
                    total_fp_count += fp_count

                    fn_seconds = self.details_dict["rec_info"][recording][segment][
                        "fn_seconds"
                    ]
                    fn_count = fn_seconds / self.segment_len
                    total_fn_count += fn_count

                    if recording in self.details_dict["true_positives"]:
                        tp_details = self.details_dict["true_positives"][recording][
                            segment
                        ]
                        tp_list = []
                        for tp in tp_details:
                            if tp[0] not in tp_list:
                                tp_list.append(tp[0])
                        tp_str = self.list_to_string(tp_list)
                    else:
                        tp_str = ""

                    if recording in self.details_dict["false_positives"]:
                        fp_details = self.details_dict["false_positives"][recording][
                            segment
                        ]
                        fp_list = []
                        for fp in fp_details:
                            if fp[0] not in fp_list:
                                fp_list.append(fp[0])
                        fp_str = self.list_to_string(fp_list)
                    else:
                        fp_str = ""

                    if recording in self.details_dict["false_negatives"]:
                        fn_list = self.details_dict["false_negatives"][recording][
                            segment
                        ]
                        fn_str = self.list_to_string(fn_list)
                    else:
                        fn_str = ""

                    file.write(
                        f"{recording},{segment},{tp_count},{fp_count},{fn_count},{tp_str},{fp_str},{fn_str}\n"
                    )

                recording_summary.append(
                    [recording, total_tp_count, total_fp_count, total_fn_count]
                )

        # write recording summary (row per recording)
        rpt_path = os.path.join(self.output_dir, "recording_summary.csv")
        df = pd.DataFrame(
            recording_summary, columns=["recording", "TP count", "FP count", "FN count"]
        )
        df.to_csv(rpt_path, index=False)

        # write details per annotated class
        rpt_path = os.path.join(self.output_dir, "classes_annotated.csv")
        with open(rpt_path, "w") as file:
            file.write(
                "class,PR-AUC,ROC-AUC,precision,recall,annotated segments,TP segments,FP segments\n"
            )
            class_precision = self.details_dict["class_precision"]
            class_recall = self.details_dict["class_recall"]
            class_valid = self.details_dict["class_valid"]
            class_invalid = self.details_dict["class_invalid"]
            class_pr_auc = self.pr_auc_dict["class_pr_auc"]
            class_roc_auc = self.roc_auc_dict["class_roc_auc"]

            segment_len = self.segment_len - self.overlap
            for i, class_code in enumerate(self.annotated_classes):
                annotations = self.y_true_annotated_df[class_code].sum()
                precision = class_precision[i]
                recall = class_recall[i]
                valid = class_valid[i] / segment_len  # convert from seconds to segments
                invalid = (
                    class_invalid[i] / segment_len
                )  # convert from seconds to segments

                if class_code in class_pr_auc:
                    pr_auc_score = class_pr_auc[class_code]
                else:
                    pr_auc_score = 0

                if class_code in class_roc_auc:
                    roc_auc_score = class_roc_auc[class_code]
                else:
                    roc_auc_score = 0

                file.write(
                    f"{class_code},{pr_auc_score:.3f},{roc_auc_score:.3f},{precision:.3f},{recall:.3f},{annotations},{valid},{invalid}\n"
                )

        # calculate and output details per non-annotated class
        classes_dict = self.get_non_annotated_class_details()
        rows = []
        for class_code in classes_dict:
            count1, count2, count3 = classes_dict[class_code]
            rows.append([class_code, count1, count2, count3])

        df = pd.DataFrame(
            rows,
            columns=[
                "class",
                "predictions >= .25",
                "predictions >= .5",
                "predictions >= .75",
            ],
        )
        df.to_csv(
            os.path.join(self.output_dir, "classes_non_annotated.csv"), index=False
        )

    def get_pr_table(self):
        """
        Calculate precision-recall table across multiple thresholds.

        This method evaluates precision and recall metrics at different threshold values
        (0.01 to 1.00 in 0.01 increments) to create comprehensive precision-recall curves.
        It calculates both per-minute granularity metrics and per-second granularity metrics.

        Returns:
            dict: Dictionary containing precision-recall data with keys:
                - annotated_thresholds: List of threshold values for annotated classes
                - annotated_precisions_minutes: List of precision values (minutes) for annotated classes
                - annotated_precisions_seconds: List of precision values (seconds) for annotated classes
                - annotated_recalls: List of recall values for annotated classes
                - trained_thresholds: List of threshold values for trained classes
                - trained_precisions: List of precision values for trained classes
                - trained_recalls: List of recall values for trained classes
                - annotated_thresholds_fine: Fine-grained thresholds for annotated classes
                - annotated_precisions_fine: Fine-grained precision values for annotated classes
                - annotated_recalls_fine: Fine-grained recall values for annotated classes
                - trained_thresholds_fine: Fine-grained thresholds for trained classes
                - trained_precisions_fine: Fine-grained precision values for trained classes
                - trained_recalls_fine: Fine-grained recall values for trained classes

        Note:
            Uses both manual threshold evaluation and scikit-learn's precision_recall_curve
            for comprehensive coverage.
        """
        from sklearn import metrics

        logging.info("Calculating PR table")
        pr_table_dict = {}
        precision, recall, thresholds = metrics.precision_recall_curve(
            self.y_true_annotated.ravel(), self.y_pred_annotated.ravel()
        )
        pr_table_dict["annotated_thresholds"] = thresholds
        pr_table_dict["annotated_precisions"] = precision
        pr_table_dict["annotated_recalls"] = recall

        precision, recall, thresholds = metrics.precision_recall_curve(
            self.y_true_trained.ravel(), self.y_pred_trained.ravel()
        )
        pr_table_dict["trained_thresholds"] = thresholds
        pr_table_dict["trained_precisions"] = precision
        pr_table_dict["trained_recalls"] = recall

        return pr_table_dict

    def initialize(self):
        """
        Initialize
        """

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        logging.info("Initializing")
        self.get_labels(self.label_dir)
        self._get_recording_info()
        self.get_annotations()

        self._init_y_true()
        self.init_y_pred(
            segments_per_recording=self.segments_per_recording, use_max_score=False
        )
        self.convert_to_numpy()

        self.y_true_annotated_df.to_csv(
            os.path.join(self.output_dir, "y_true_annotated.csv"), index=False
        )
        self.y_pred_annotated_df.to_csv(
            os.path.join(self.output_dir, "y_pred_annotated.csv"), index=False
        )
        self.y_true_trained_df.to_csv(
            os.path.join(self.output_dir, "y_true_trained.csv"), index=False
        )
        self.y_pred_trained_df.to_csv(
            os.path.join(self.output_dir, "y_pred_trained.csv"), index=False
        )
        self.check_if_arrays_match()

    def plot_calibration_curve(self, y_true, y_pred, a, b, n_bins=15):
        """
        Plot calibration curve comparing uncalibrated and Platt-calibrated predictions.

        This method creates a reliability diagram (calibration curve) that shows how well
        the model's predicted probabilities match the observed frequencies. It plots both
        the original uncalibrated predictions and the Platt-scaled calibrated predictions
        against the ideal calibration line.

        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Uncalibrated model probabilities
            a: Platt scaling coefficient
            b: Platt scaling intercept
            n_bins: Number of bins for the calibration curve (default: 15)

        Note:
            Saves the calibration plot to the output directory with filename format:
            calibration-{a:.2f}-{b:.2f}.png
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.special import expit, logit
        from sklearn.calibration import calibration_curve

        if self.coefficient is not None:
            # plot with the values specified in the constructor
            a = self.coefficient
            b = self.intercept

        output_path = str(Path(self.output_dir) / f"calibration-{a:.2f}-{b:.2f}.png")

        # Apply Platt scaling to model probabilities
        logits = logit(np.clip(y_pred, 1e-7, 1 - 1e-7))
        calibrated_pred = expit(a * logits + b)

        # Compute calibration curves
        frac_pos_uncal, mean_pred_uncal = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy="uniform"
        )
        frac_pos_cal, mean_pred_cal = calibration_curve(
            y_true, calibrated_pred, n_bins=n_bins, strategy="uniform"
        )

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(mean_pred_uncal, frac_pos_uncal, label="Uncalibrated", marker="o")
        plt.plot(mean_pred_cal, frac_pos_cal, label="Platt calibrated", marker="o")
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated"
        )

        plt.xlabel("Predicted probability", fontsize=20)
        plt.ylabel("Observed frequency", fontsize=20)
        plt.title("Calibration Curve (Reliability Diagram)", fontsize=26)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logging.info(f"Saved calibration plot to {output_path}")

    def do_calibration(self):
        """
        Calculate and print optimal Platt scaling coefficient and intercept.

        This method performs Platt scaling calibration on the model predictions to improve
        probability calibration. It uses logistic regression to find optimal scaling parameters
        that transform the raw logits to better-calibrated probabilities.

        The method filters predictions above the calibration cutoff threshold, fits a
        logistic regression model to the logits, and outputs the optimal coefficient and
        intercept values. It also generates a calibration curve plot for visualization.

        Note:
            Requires that self.y_pred_trained and self.y_true_trained have been initialized.
            The calibration_cutoff parameter controls which predictions are used for fitting.
            Generates a calibration plot saved to the output directory.

        Raises:
            ValueError: If too few samples are above the calibration cutoff threshold.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from scipy.special import logit

        y_pred = self.y_pred_trained.ravel()
        y_true = self.y_true_trained.ravel()

        # convert predictions back to logits
        eps = 1e-7
        logits = logit(np.clip(y_pred, eps, 1 - eps))

        # filter out predictions below the cutoff
        mask = y_pred >= self.calibration_cutoff
        if np.sum(mask) < 10:
            raise ValueError(
                "Too few samples above fit_threshold to fit Platt scaling."
            )

        # use logistic regression to find optimal scaling parameters
        X = logits[mask].reshape(-1, 1)
        y = y_true[mask]
        platt_model = LogisticRegression(solver="lbfgs")
        platt_model.fit(X, y)

        coefficient = platt_model.coef_[0][0]
        intercept = platt_model.intercept_[0]

        # plot a curve
        self.plot_calibration_curve(y_true, y_pred, coefficient, intercept)

        # print the coefficient and intercept
        logging.info(f"Coefficient = {coefficient:.2f}")
        logging.info(f"Intercept   = {intercept:.2f}")

    def run(self):
        """
        Execute the complete testing workflow.

        This method orchestrates the entire testing process by:
        1. Initializing the tester and loading data
        2. If calibrate=True, performing calibration analysis and returning early
        3. Calculating PR-AUC (Precision-Recall Area-Under-Curve) statistics
        4. Calculating ROC-AUC (Receiver Operating Characteristic Area-Under-Curve) statistics
        5. Calculating precision-recall statistics at the specified threshold
        6. Generating a precision-recall table across multiple thresholds
        7. Producing comprehensive output reports

        The method calls all necessary setup, calculation, and reporting methods
        in the correct order to complete the analysis workflow.

        Note:
            This is the main entry point for running a complete test analysis.
            If calibrate=True, only calibration analysis is performed and the method returns early.
            All output files will be written to self.output_dir.
        """

        self.initialize()

        if self.calibrate:
            self.do_calibration()
            return

        # calculate stats
        logging.info("Calculating PR-AUC stats")
        self.pr_auc_dict = self.get_pr_auc_stats()

        logging.info("Calculating ROC stats")
        self.roc_auc_dict = self.get_roc_auc_stats()

        logging.info("Calculating PR stats")
        self.details_dict = self.get_precision_recall(
            threshold=self.threshold, details=True
        )
        self.pr_table_dict = self.get_pr_table()

        logging.info(f"Creating reports in {self.output_dir}")
        self._produce_reports()
