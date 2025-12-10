# Defer some imports to improve initialization performance.
import logging
import os

from britekit.core.config_loader import get_config
from britekit.testing.base_tester import BaseTester


class PerRecordingTester(BaseTester):
    """
    Calculate test metrics when annotations are specified per recording. That is, the ground truth data
    gives a list of classes per recording, with no indication of where in the recording they are heard.
    This has the advantage that new tests can be created very quickly. By assuming that all detections
    of a valid class are valid, we can count the number of TP and FP seconds. However, FNs can only be
    counted at the recording level, so our recall measure is extremely coarse. To work around this, we can
    output the number of TP seconds at a given precision, say 95%. Given two tests, this can be used to
    measure relative but not absolute recall.

    Annotations are defined in a CSV with two columns: "recording", and "classes".
    The recording column is the file name without the path or type suffix, e.g. "recording1".
    The classes column contains a comma-separated list of codes for the classes found in the corresponding
    recording. If your annotations are in a different format, simply convert to this format to use this script.

    Classifiers should be run with a threshold of 0, and with label merging disabled so segment-specific scores are retained.

    Attributes:
        annotation_path (str): Annotations CSV file.
        recording_dir (str): Directory containing recordings.
        label_dir (str): Directory containing Audacity labels.
        output_dir (str): Output directory, where reports will be written.
        threshold (float): Score threshold for precision/recall reporting.
        tp_secs_at_precision (float, optional): Granular recall cannot be calculated with per-recording annotations,
        but reporting TP seconds at this precision is a useful proxy (default=.95).
    """

    def __init__(
        self,
        annotation_path: str,
        recording_dir: str,
        label_dir: str,
        output_dir: str,
        threshold: float,
        tp_secs_at_precision: float = 0.95,
    ):
        """
        Initialize the PerRecordingTester.

        See class docstring for detailed parameter descriptions and usage information.
        """
        super().__init__()
        self.annotation_path = annotation_path
        self.recording_dir = recording_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.tp_secs_at_precision = tp_secs_at_precision
        self.per_recording = True

        self.cfg = get_config()

    # ============================================================================
    # Public methods - Main execution
    # ============================================================================

    def run(self):
        """
        Execute the complete testing workflow.

        This method orchestrates the entire testing process by:
        1. Initializing the tester and loading data
        2. Calculating PR-AUC (Precision-Recall Area-Under-Curve) statistics
        3. Calculating ROC-AUC (Receiver Operating Characteristic Area-Under-Curve) statistics
        4. Calculating precision-recall statistics at the specified threshold
        5. Generating a precision-recall table across multiple thresholds
        6. Producing comprehensive output reports

        The method calls all necessary setup, calculation, and reporting methods
        in the correct order to complete the analysis workflow.

        Note:
            This is the main entry point for running a complete test analysis.
            All output files will be written to self.output_dir.
        """

        self._initialize()

        # calculate stats
        logging.info("Calculating PR-AUC stats")
        self.map_dict = self.get_pr_auc_stats()

        logging.info("Calculating ROC-AUC stats")
        self.roc_dict = self.get_roc_auc_stats()

        logging.info("Calculating PR stats")
        self.details_dict = self.get_precision_recall(
            threshold=self.threshold, details=True
        )

        logging.info("Calculating PR table")
        self.pr_table_dict = self.get_pr_table()

        logging.info(f"Creating reports in {self.output_dir}")
        self.produce_reports()

    # ============================================================================
    # Public methods - Data loading and processing
    # ============================================================================

    def get_annotations(self):
        """
        Load annotation data from CSV file and process into internal format.

        This method reads a CSV file containing ground truth annotations where each row
        represents a recording and its associated classes. The CSV should have columns:
        "recording" (filename without path/extension) and "classes" (comma-separated class codes).

        The method processes the annotations, handles class code mapping, filters out
        unknown classes, and organizes the data for subsequent analysis.

        Note:
            Sets self.annotations, self.annotated_class_set, and self.annotated_classes.
            Calls self.set_class_indexes() to update class indexing.
        """
        import pandas as pd

        unknown_classes = set()
        self.annotations = {}
        self.annotated_class_set = set()
        df = pd.read_csv(
            self.annotation_path,
            dtype={"recording": str, "classes": str},
            keep_default_na=False,
        )
        for i, row in df.iterrows():
            recording = row["recording"]
            if not recording:
                break

            if recording not in self.annotations:
                self.annotations[recording] = []

            input_class_list = []
            for code in row["classes"].split(","):
                input_class_list.append(code.strip())

            for class_code in input_class_list:
                if class_code not in self.trained_class_set:
                    if (
                        self.cfg.misc.map_codes
                        and class_code in self.cfg.misc.map_codes
                    ):
                        class_code = self.cfg.misc.map_codes[class_code]
                    elif len(class_code) > 0:
                        # the unknown_classes set is just so we only report each unknown class once
                        if class_code not in unknown_classes:
                            logging.error(
                                f"Unknown class {class_code} will be skipped (is it in ignore list?)."
                            )
                            unknown_classes.add(class_code)

                        continue  # exclude from saved annotations

                if class_code:
                    self.annotations[recording].append(class_code)
                    self.annotated_class_set.add(class_code)

        self.annotated_classes = sorted(list(self.annotated_class_set))
        self.set_class_indexes()

    # ============================================================================
    # Public methods - Statistics and metrics
    # ============================================================================

    def get_pr_table(self):
        """
        Calculate precision-recall table across multiple thresholds.

        This method evaluates precision and recall metrics at different threshold values
        (0.01 to 1.00 in 0.01 increments) to create a comprehensive precision-recall curve.
        It calculates both per-recording granularity metrics and per-second granularity metrics.

        Returns:
            dict: Dictionary containing precision-recall data with keys:
                - precisions: List of precision values at each threshold
                - recalls: List of recall values at each threshold
                - precision_secs: List of precision values in seconds at each threshold
                - tp_secs: List of true positive seconds at each threshold
                - fp_secs: List of false positive seconds at each threshold
                - thresholds: List of threshold values used

        Note:
            Rows with precision=0 at the end are trimmed to avoid unnecessary data points.
        """
        import numpy as np

        precisions = []  # precision in segments or recordings
        recalls = []
        precision_secs = []  # precision in seconds
        tp_secs = []
        fp_secs = []
        thresholds = []
        for threshold in np.arange(0.01, 1.01, 0.01):
            pr_dict = self.get_precision_recall(threshold, details=False)
            precisions.append(pr_dict["precision_annotated"])
            recalls.append(pr_dict["recall_annotated"])
            precision_secs.append(pr_dict["precision_secs"])
            tp_secs.append(pr_dict["tp_secs"])
            fp_secs.append(pr_dict["fp_secs"])
            thresholds.append(threshold)

        # trim any rows with precision=0 at the end
        trim_num = 0
        for i in range(len(precisions) - 1, -1, -1):
            if precisions[i] == 0:
                trim_num += 1

        if trim_num > 0:
            precisions = precisions[:-trim_num]
            recalls = recalls[:-trim_num]
            precision_secs = precision_secs[:-trim_num]
            tp_secs = tp_secs[:-trim_num]
            fp_secs = fp_secs[:-trim_num]
            thresholds = thresholds[:-trim_num]

        ret_dict = {}
        ret_dict["precisions"] = precisions
        ret_dict["recalls"] = recalls
        ret_dict["precision_secs"] = precision_secs
        ret_dict["tp_secs"] = tp_secs
        ret_dict["fp_secs"] = fp_secs
        ret_dict["thresholds"] = thresholds

        return ret_dict

    # ============================================================================
    # Public methods - Report generation
    # ============================================================================

    def produce_reports(self):
        """
        Generate comprehensive output reports and CSV files.

        This method creates multiple output files containing detailed analysis results:
        - Precision-recall table and curve data
        - Summary report with key metrics
        - Recording-level details and summaries
        - Class-level performance statistics

        The method generates the following files in the output directory:
        - pr_table.csv: Precision-recall data at different thresholds
        - pr_curve.csv: Interpolated precision-recall curve
        - summary_report.txt: Human-readable summary with key metrics
        - recording_details.csv: Detailed statistics per recording
        - recording_summary.csv: Summary statistics per recording
        - class.csv: Performance metrics per class

        Note:
            Requires that self.map_dict, self.roc_dict, self.details_dict, and
            self.pr_table_dict have been calculated by calling the corresponding methods.
        """
        import numpy as np
        import pandas as pd

        # calculate and output precision/recall per threshold
        threshold = self.pr_table_dict["thresholds"]
        precision = self.pr_table_dict["precisions"]
        recall = self.pr_table_dict["recalls"]

        df = pd.DataFrame()
        df["threshold"] = threshold
        df["precision"] = precision
        df["recall"] = recall
        df.to_csv(
            os.path.join(self.output_dir, "pr_table.csv"),
            index=False,
            float_format="%.3f",
        )

        # convert that to recall per precision
        interpolated_precision, interpolated_recall = self.interpolate(
            precision, recall
        )
        df = pd.DataFrame()
        df["precision"] = interpolated_precision
        df["recall"] = interpolated_recall
        df.to_csv(
            os.path.join(self.output_dir, "pr_curve.csv"),
            index=False,
            float_format="%.3f",
        )

        # get TP seconds at specified precision
        i = np.searchsorted(np.array(precision), self.tp_secs_at_precision)
        if i > 0:
            report_tp_secs = self.pr_table_dict["tp_secs"][i]
        else:
            report_tp_secs = 0

        # output a summary report
        rpt = []
        rpt.append(
            f"TP seconds at precision {self.tp_secs_at_precision:.2f} = {report_tp_secs} # using fine-grained precision metric\n\n"
        )
        rpt.append(
            "Remaining metrics use per-recording granularity, which is of questionable\n"
        )
        rpt.append(
            "value, especially if the recordings are of different durations.\n\n"
        )
        rpt.append(
            f"Macro-averaged PR-AUC score = {self.map_dict['macro_pr_auc']:.4f}\n"
        )
        rpt.append(
            f"Micro-averaged PR-AUC score = {self.map_dict['micro_pr_auc_annotated']:.4f}\n"
        )

        rpt.append(
            f"Macro-averaged ROC-AUC score = {self.roc_dict['macro_roc_auc']:.4f}\n"
        )
        rpt.append(
            f"Micro-averaged ROC-AUC score = {self.roc_dict['micro_roc_auc_annotated']:.4f}\n"
        )

        rpt.append(f"Details for threshold = {self.threshold}:\n")
        rpt.append(
            f"   Precision (recording) = {100 * self.details_dict['precision_annotated']:.2f}%\n"
        )
        rpt.append(
            f"   Precision (seconds) = {100 * self.details_dict['precision_secs']:.2f}% # the only fine-grained metric in this section\n"
        )
        rpt.append(
            f"   Recall (recording) = {100 * self.details_dict['recall_annotated']:.2f}%\n"
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
                "directory,recording,TP count,FP count,FN count,TP classes,FP classes,FN classes\n"
            )
            for recording in self.details_dict["rec_info"]:
                rec_info = self.details_dict["rec_info"][recording]

                tp_seconds = rec_info["tp_seconds"]
                tp_count = tp_seconds / self.segment_len

                fp_seconds = rec_info["fp_seconds"]
                fp_count = fp_seconds / self.segment_len

                fn_seconds = rec_info["fn_seconds"]
                fn_count = fn_seconds / self.segment_len

                if recording in self.details_dict["true_positives"]:
                    tp_details = self.details_dict["true_positives"][recording]
                    tp_list = []
                    for tp in tp_details:
                        if tp[0] not in tp_list:
                            tp_list.append(tp[0])
                    tp_str = self.list_to_string(tp_list)
                else:
                    tp_str = ""

                if recording in self.details_dict["false_positives"]:
                    fp_details = self.details_dict["false_positives"][recording]
                    fp_list = []
                    for fp in fp_details:
                        if fp[0] not in fp_list:
                            fp_list.append(fp[0])
                    fp_str = self.list_to_string(fp_list)
                else:
                    fp_str = ""

                if recording in self.details_dict["false_negatives"]:
                    fn_list = self.details_dict["false_negatives"][recording]
                    fn_str = self.list_to_string(fn_list)
                else:
                    fn_str = ""

                file.write(
                    f"{self.recording_dir},{recording},{tp_count},{fp_count},{fn_count},{tp_str},{fp_str},{fn_str}\n"
                )

                recording_summary.append(
                    [self.recording_dir, recording, tp_count, fp_count, fn_count]
                )

        # write recording summary (row per recording)
        rpt_path = os.path.join(self.output_dir, "recording_summary.csv")
        df = pd.DataFrame(
            recording_summary,
            columns=["directory", "recording", "TP count", "FP count", "FN count"],
        )
        df.to_csv(rpt_path, index=False)

        # write details per class
        rpt_path = os.path.join(self.output_dir, "class.csv")
        with open(rpt_path, "w") as file:
            file.write(
                "class,PR-AUC,ROC-AUC,precision,recall,annotated recordings,TP seconds,FP seconds\n"
            )
            class_precision = self.details_dict["class_precision"]
            class_recall = self.details_dict["class_recall"]
            class_valid = self.details_dict["class_valid"]
            class_invalid = self.details_dict["class_invalid"]
            class_pr_auc = self.map_dict["class_pr_auc"]
            class_roc_auc = self.roc_dict["class_roc_auc"]

            for i, class_code in enumerate(self.annotated_classes):
                annotations = self.y_true_annotated_df[class_code].sum()
                precision = class_precision[i]
                recall = class_recall[i]
                valid = class_valid[i]
                invalid = class_invalid[i]

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

    # ============================================================================
    # Private methods - Initialization and setup
    # ============================================================================

    def _initialize(self):
        """
        Initialize
        """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # initialize y_true and y_pred and save them as CSV files
        logging.info("Initializing")
        self.get_labels(self.label_dir)
        self.get_annotations()
        self._init_y_true()
        self._init_y_pred()
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

    def _init_y_true(self):
        """
        Create a dataframe representing the ground truth data, with recordings segmented into segments
        """
        import pandas as pd

        # convert self.annotations to 2D array with a row per segment and a column per class;
        # set cells to 1 if class is present and 0 if not present
        self.recordings = []  # base class needs array with recording per row
        rows = []
        for recording in sorted(self.annotations.keys()):
            self.recordings.append(recording)
            row = [recording]
            row.extend([0 for class_code in self.trained_classes])
            for class_code in self.annotations[recording]:
                if class_code in self.trained_class_indexes:
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

    def _init_y_pred(self):
        """
        Create y_pred dataframe with per-recording granularity
        """
        import pandas as pd

        rows = []
        for i, recording in enumerate(sorted(self.labels_per_recording.keys())):
            row = [0 for class_code in self.trained_classes]
            for label in self.labels_per_recording[recording]:
                if label.class_code not in self.trained_class_indexes:
                    continue

                # use max so we use the highest score for this class in this recording
                row[self.trained_class_indexes[label.class_code]] = max(
                    row[self.trained_class_indexes[label.class_code]], label.score
                )

            rows.append([recording] + row)

        self.y_pred_trained_df = pd.DataFrame(rows, columns=[""] + self.trained_classes)

        # create version for annotated classes only
        self.y_pred_annotated_df = self.y_pred_trained_df.copy()
        for i, column in enumerate(self.y_pred_annotated_df.columns):
            if i == 0:
                continue  # skip the index column

            if column not in self.annotated_class_set:
                self.y_pred_annotated_df = self.y_pred_annotated_df.drop(column, axis=1)
