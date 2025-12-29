#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import logging
import os


from britekit.core.config_loader import get_config
from britekit.testing.base_tester import BaseTester


class Annotation:
    def __init__(self, start_time, end_time, class_code):
        self.start_time = start_time
        self.end_time = end_time
        self.class_code = class_code

    def __str__(self):
        return f"{self.class_code}: {self.start_time}-{self.end_time}"


class PerBlockTester(BaseTester):
    """
    Calculate test metrics when annotations are specified per block, where a block is a fixed length, such
    as a minute. That is, for selected blocks of each recording, a list of classes known to be present is given,
    and we are to calculate metrics for those blocks only.

    Annotations are read as a CSV with three columns: "recording", "block", and "classes".
    The recording column is the file name without the path or type suffix, e.g. "recording1".
    The block column contains 1 for the first block, 2 for the second block etc. and may exclude some blocks.
    The classes column contains a comma-separated list of codes for the classes found in the corresponding block.
    If your annotations are in a different format, simply convert to this format to use this script.

    Classifiers should be run with a threshold of 0, and with label merging disabled so segment-specific scores are retained.

    Attributes:
        annotation_path (str): Annotations CSV file.
        recording_dir (str): Directory containing recordings.
        label_dir (str): Directory containing Audacity labels.
        output_dir (str): Output directory, where reports will be written.
        threshold (float): Score threshold for precision/recall reporting.
        block_size (int, optional): block_size in seconds (default=60).
        gen_pr_table (bool, optional): If true, generate a PR table, which may be slow (default = False).
    """

    def __init__(
        self,
        annotation_path: str,
        recording_dir: str,
        label_dir: str,
        output_dir: str,
        threshold: float,
        block_size: int = 60,
        gen_pr_table: bool = False,
    ):
        """
        Initialize the PerBlockTester.

        See class docstring for detailed parameter descriptions and usage information.
        """
        super().__init__()
        self.annotation_path = annotation_path
        self.recording_dir = recording_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.block_size = block_size
        self.gen_pr_table = gen_pr_table

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
        5. Generating a precision-recall table across multiple thresholds (if gen_pr_table=True)
        6. Producing comprehensive output reports

        The method calls all necessary setup, calculation, and reporting methods
        in the correct order to complete the analysis workflow.

        Note:
            This is the main entry point for running a complete test analysis.
            All output files will be written to self.output_dir.
            The gen_pr_table parameter controls whether detailed PR table analysis is performed.
        """

        self._initialize()

        # calculate stats
        logging.info("Calculating PR-AUC stats")
        self.pr_auc_dict = self.get_pr_auc_stats()

        logging.info("Calculating ROC-AUC stats")
        self.roc_auc_dict = self.get_roc_auc_stats()

        logging.info("Calculating PR stats")
        self.details_dict = self.get_precision_recall(
            threshold=self.threshold, details=True
        )

        if self.gen_pr_table:
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
        represents a recording, block, and its associated classes. The CSV should have columns:
        "recording" (filename without path/extension), "block" (block number starting from 1),
        and "classes" (comma-separated class codes).

        The method processes the annotations, handles class code mapping, filters out
        unknown classes, and organizes the data for subsequent analysis.

        Note:
            Sets self.annotations, self.annotated_class_set, self.annotated_classes,
            and self.segments_per_recording. Calls self.set_class_indexes() to update class indexing.
        """
        import pandas as pd

        # read the annotations
        unknown_classes = set()
        self.annotations = {}
        self.annotated_class_set = set()
        self.segments_per_recording = {}
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
                self.annotations[recording] = {}
                self.segments_per_recording[recording] = []

            block = row["block"]
            if block not in self.annotations[recording]:
                self.annotations[recording][block] = []
                self.segments_per_recording[recording].append(block - 1)

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
                    self.annotations[recording][block].append(class_code)
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
        (0.01 to 1.00 in 0.01 increments) to create comprehensive precision-recall curves.
        It calculates both per-block granularity metrics and per-second granularity metrics.

        Returns:
            dict: Dictionary containing precision-recall data with keys:
                - annotated_thresholds: List of threshold values for annotated classes
                - annotated_precisions_blocks: List of precision values (blocks) for annotated classes
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
        import numpy as np
        from sklearn import metrics

        # use the looping method so we get per_second precision
        thresholds = []
        recall_annotated, precision_annotated_blocks, precision_annotated_seconds = (
            [],
            [],
            [],
        )
        recall_trained, precision_trained_blocks = [], []
        for threshold in np.arange(0.01, 1.01, 0.01):
            info = self.get_precision_recall(threshold)
            thresholds.append(threshold)
            recall_annotated.append(info["recall_annotated"])
            precision_annotated_blocks.append(info["precision_annotated"])
            precision_annotated_seconds.append(info["precision_secs"])
            recall_trained.append(info["recall_trained"])
            precision_trained_blocks.append(info["precision_trained"])
            logging.info(
                f"\rPercent complete: {int(threshold * 100)}%", end="", flush=True
            )

        logging.info("")
        pr_table_dict = {}
        pr_table_dict["annotated_thresholds"] = thresholds
        pr_table_dict["annotated_precisions_blocks"] = precision_annotated_blocks
        pr_table_dict["annotated_precisions_seconds"] = precision_annotated_seconds
        pr_table_dict["annotated_recalls"] = recall_annotated

        pr_table_dict["trained_thresholds"] = thresholds
        pr_table_dict["trained_precisions"] = precision_trained_blocks
        pr_table_dict["trained_recalls"] = recall_trained

        # use this method for more granular results without per_second precision
        precision, recall, thresholds = metrics.precision_recall_curve(
            self.y_true_annotated.ravel(), self.y_pred_annotated.ravel()
        )
        pr_table_dict["annotated_thresholds_fine"] = thresholds
        pr_table_dict["annotated_precisions_fine"] = precision
        pr_table_dict["annotated_recalls_fine"] = recall

        precision, recall, thresholds = metrics.precision_recall_curve(
            self.y_true_trained.ravel(), self.y_pred_trained.ravel()
        )
        pr_table_dict["trained_thresholds_fine"] = thresholds
        pr_table_dict["trained_precisions_fine"] = precision
        pr_table_dict["trained_recalls_fine"] = recall

        return pr_table_dict

    # ============================================================================
    # Public methods - Report generation
    # ============================================================================

    def produce_reports(self):
        """
        Generate comprehensive output reports and CSV files.

        This method creates multiple output files containing detailed analysis results:
        - Precision-recall tables and curves (if gen_pr_table=True)
        - ROC-AUC curves and analysis
        - Summary report with key metrics
        - Recording-level details and summaries
        - Class-level performance statistics
        - Prediction range distribution analysis

        The method generates the following files in the output directory:
        - pr_per_threshold_*.csv/png: Precision-recall data at different thresholds
        - pr_curve_*.csv/png: Precision-recall curves
        - roc_*.csv/png: ROC-AUC curve analysis
        - summary_report.txt: Human-readable summary with key metrics
        - recording_details_trained.csv: Detailed statistics per recording/segment
        - recording_summary_trained.csv: Summary statistics per recording
        - class_annotated.csv: Performance metrics per annotated class
        - class_non_annotated.csv: Prediction statistics for non-annotated classes
        - prediction_range_counts.csv: Distribution of prediction scores

        Note:
            Requires that self.map_dict, self.roc_dict, self.details_dict, and
            self.pr_table_dict (if gen_pr_table=True) have been calculated by calling
            the corresponding methods.
        """
        import numpy as np
        import pandas as pd

        if self.gen_pr_table:
            # calculate and output precision/recall per threshold
            threshold_annotated = self.pr_table_dict["annotated_thresholds"]
            precision_annotated = self.pr_table_dict["annotated_precisions_blocks"]
            precision_annotated_secs = self.pr_table_dict[
                "annotated_precisions_seconds"
            ]
            recall_annotated = self.pr_table_dict["annotated_recalls"]
            self._output_pr_per_threshold(
                threshold_annotated,
                precision_annotated,
                recall_annotated,
                "pr_per_threshold_annotated",
                precision_annotated_secs,
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
            precision_annotated_fine = self.pr_table_dict["annotated_precisions_fine"]
            recall_annotated_fine = self.pr_table_dict["annotated_recalls_fine"]
            self._output_roc_curves(
                roc_thresholds,
                roc_tpr,
                roc_fpr,
                precision_annotated_fine,
                recall_annotated_fine,
                "annotated",
            )

            roc_thresholds = self.roc_auc_dict["roc_thresholds_trained"]
            roc_tpr = self.roc_auc_dict["roc_tpr_trained"]
            roc_fpr = self.roc_auc_dict["roc_fpr_trained"]
            precision_trained_fine = self.pr_table_dict["trained_precisions_fine"]
            recall_trained_fine = self.pr_table_dict["trained_recalls_fine"]
            self._output_roc_curves(
                roc_thresholds,
                roc_tpr,
                roc_fpr,
                precision_trained_fine,
                recall_trained_fine,
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
            f"      Precision (blocks) = {100 * self.details_dict['precision_annotated']:.2f}%\n"
        )
        rpt.append(
            f"      Precision (seconds) = {100 * self.details_dict['precision_secs']:.2f}%\n"
        )
        rpt.append(
            f"      Recall (blocks) = {100 * self.details_dict['recall_annotated']:.2f}%\n"
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
            f"      Precision (blocks) = {100 * self.details_dict['precision_trained']:.2f}%\n"
        )
        rpt.append(
            f"      Recall (blocks) = {100 * self.details_dict['recall_trained']:.2f}%\n"
        )
        logging.info("")
        with open(os.path.join(self.output_dir, "summary_report.txt"), "w") as summary:
            for rpt_line in rpt:
                logging.info(rpt_line[:-1])
                summary.write(rpt_line)

        # write recording details (row per segment)
        recording_summary = []
        rpt_path = os.path.join(self.output_dir, "recording_details_trained.csv")
        with open(rpt_path, "w") as file:
            file.write(
                "recording,segment,TP count,FP count,FN count,TP class,FP class,FN class\n"
            )
            for recording in self.details_dict["rec_info"]:
                for i, segment in enumerate(self.segments_per_recording[recording]):
                    tp_count = 0
                    if recording in self.details_dict["true_positives"]:
                        tp_details = self.details_dict["true_positives"][recording][i]
                        tp_list = []
                        for tp in tp_details:
                            if tp[0] not in tp_list:
                                tp_list.append(tp[0])
                                tp_count += 1
                        tp_str = self.list_to_string(tp_list)
                    else:
                        tp_str = ""

                    fp_count = 0
                    if recording in self.details_dict["false_positives"]:
                        fp_details = self.details_dict["false_positives"][recording][i]
                        fp_list = []
                        for fp in fp_details:
                            if fp[0] not in fp_list:
                                fp_list.append(fp[0])
                                fp_count += 1
                        fp_str = self.list_to_string(fp_list)
                    else:
                        fp_str = ""

                    fn_count = 0
                    if recording in self.details_dict["false_negatives"]:
                        fn_list = self.details_dict["false_negatives"][recording][i]
                        fn_count = len(fn_list)
                        fn_str = self.list_to_string(fn_list)
                    else:
                        fn_str = ""

                    file.write(
                        f"{recording},{segment + 1},{tp_count},{fp_count},{fn_count},{tp_str},{fp_str},{fn_str}\n"
                    )

                recording_summary.append([recording, tp_count, fp_count, fn_count])

        # write recording summary (row per recording)
        rpt_path = os.path.join(self.output_dir, "recording_summary_trained.csv")
        df = pd.DataFrame(
            recording_summary, columns=["recording", "TP count", "FP count", "FN count"]
        )
        df.to_csv(rpt_path, index=False)

        # write details per annotated class
        rpt_path = os.path.join(self.output_dir, "class_annotated.csv")
        with open(rpt_path, "w") as file:
            file.write(
                "class,PR-AUC,ROC-AUC,precision,recall,annotated segments,TP seconds,FP seconds\n"
            )
            class_precision = self.details_dict["class_precision"]
            class_recall = self.details_dict["class_recall"]
            class_valid = self.details_dict["class_valid"]
            class_invalid = self.details_dict["class_invalid"]
            class_pr_auc = self.pr_auc_dict["class_pr_auc"]
            class_roc_auc = self.roc_auc_dict["class_roc_auc"]

            for i, class_code in enumerate(self.annotated_classes):
                annotations = self.y_true_annotated_df[class_code].sum()
                precision = class_precision[i]
                recall = class_recall[i]
                valid = class_valid[i]  # TP seconds
                invalid = class_invalid[i]  # FP seconds

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
        class_dict = self.get_non_annotated_class_details()
        rows = []
        for class_code in class_dict:
            count1, count2, count3 = class_dict[class_code]
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
        df.to_csv(os.path.join(self.output_dir, "class_non_annotated.csv"), index=False)

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
        self.get_labels(self.label_dir, segment_len=self.block_size, overlap=0)
        self.get_annotations()
        self._init_y_true()
        self.init_y_pred(segments_per_recording=self.segments_per_recording)
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
        Create a dataframe representing the ground truth data, with recordings segmented into 1-block segments
        """
        import pandas as pd

        # convert self.annotations to 2D array with a row per segment and a column per class;
        # set cells to 1 if class is present and 0 if not present
        self.recordings = []  # base class needs array with recording per row
        rows = []
        for recording in sorted(self.annotations.keys()):
            for block in sorted(self.annotations[recording].keys()):
                self.recordings.append(recording)
                row = [f"{recording}-{block - 1}"]
                row.extend([0 for class_code in self.trained_classes])
                for class_code in self.annotations[recording][block]:
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

    # ============================================================================
    # Private methods - Report generation utilities
    # ============================================================================

    def _output_pr_per_threshold(
        self, threshold, precision, recall, name, precision_secs=None
    ):
        """
        Output precision/recall per threshold
        """
        from matplotlib import pyplot as plt
        import pandas as pd

        df = pd.DataFrame()
        df["threshold"] = pd.Series(threshold)
        df["recall (blocks)"] = pd.Series(recall)
        df["precision (blocks)"] = pd.Series(precision)
        if precision_secs is not None:
            df["precision (seconds)"] = pd.Series(precision_secs)

        df.to_csv(
            os.path.join(self.output_dir, f"{name}.csv"),
            index=False,
            float_format="%.3f",
        )

        plt.clf()
        plt.plot(recall, label="Recall")
        plt.plot(precision, label="Precision (blocks)")
        if precision_secs is not None:
            plt.plot(precision_secs, label="Precision (Seconds)")

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
        from matplotlib import pyplot as plt
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
        from matplotlib import pyplot as plt
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
