# Defer some imports to improve initialization performance.
import copy
import logging
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Optional

from britekit.core.config_loader import get_config
from britekit.core.exceptions import InputError
from britekit.core import util


def natural_key(s):
    """
    Key used in sorting training log directories.
    Ensure that "version_2" sorts before "version_19" etc.
    """
    return int(re.search(r"\d+", s).group())


class Tuner:
    """
    Tune the joint values of selected hyperparameters, either by exhaustive or random search.
    """

    def __init__(
        self,
        recording_dir: str,
        output_dir: str,
        annotation_path: str,
        train_log_dir: str,
        metric: str,
        param_space: Optional[Any],
        num_trials: int = 0,
        num_runs: int = 1,
        extract: bool = False,
        skip_training: bool = False,
        classes_path: Optional[str] = None,
    ):
        from britekit.core.pickler import TrainingPickler
        from britekit.core.reextractor import Reextractor

        self.cfg = get_config()
        self.original_seed = self.cfg.train.seed
        self.recording_dir = recording_dir
        self.output_dir = output_dir
        self.annotation_path = annotation_path
        self.train_log_dir = train_log_dir
        self.param_space = param_space
        self.num_trials = num_trials
        self.num_runs = num_runs
        self.extract = extract
        self.skip_training = skip_training
        self.classes_path = classes_path

        if self.extract:
            self.spec_group = "__temp__"
            self.reextractor = Reextractor(
                db_path=self.cfg.train.train_db,
                class_name=None,
                classes_path=self.classes_path,
                check=False,
                spec_group=self.spec_group,
            )

            self.pickle_file = tempfile.NamedTemporaryFile(mode="wb")
            self.pickler = TrainingPickler(
                db_path=self.cfg.train.train_db,
                output_path=self.pickle_file.name,
                classes_path=self.classes_path,
                max_per_class=None,
                spec_group=self.spec_group,
            )
            self.cfg.train.train_pickle = self.pickle_file.name

        # map short metric name to full name
        metric_dict = {
            "macro_pr": "macro_pr_auc",
            "micro_pr": "micro_pr_auc_trained",
            "macro_roc": "macro_roc_auc",
            "micro_roc": "micro_roc_auc_trained",
        }

        if metric not in metric_dict:
            raise InputError(f"Invalid metric: {metric}")

        self.metric = metric_dict[metric]
        logging.info(f"Using metric {metric}")

    def _get_values(self, param_def):
        """
        Return list of possible values for a hyperparameter definition.
        """
        if param_def["type"] == "categorical":
            return param_def["choices"]
        else:
            if param_def["bounds"][0] > param_def["bounds"][1]:
                raise ValueError(f"Invalid bounds in {param_def}")

            if param_def["type"] == "int":
                return [
                    i
                    for i in range(
                        param_def["bounds"][0],
                        param_def["bounds"][1] + 1,
                        param_def["step"],
                    )
                ]
            elif param_def["type"] == "float":
                return util.get_range(
                    param_def["bounds"][0], param_def["bounds"][1], param_def["step"]
                )
            else:
                raise ValueError(f"Unknown param type: {param_def['type']}")

    def _set_value(self, param_def, value):
        """
        Update configuration with specified hyperparameter value.
        """
        name = param_def["name"]

        logging.info(f"*** Set {name}={value}")
        if hasattr(self.cfg.train, name):
            setattr(self.cfg.train, name, value)
        elif hasattr(self.cfg.audio, name):
            setattr(self.cfg.audio, name, value)
        elif hasattr(self.cfg.infer, name):
            setattr(self.cfg.infer, name, value)
        else:
            # no main attribute of that name, so check augmentations
            found = False
            for aug in self.cfg.train.augmentations:
                if aug["name"] == name:
                    aug["prob"] = value
                    found = True
                    break

            if not found:
                # no augmentation of that name, so check their sub-parameters
                for aug in self.cfg.train.augmentations:
                    for key in aug["params"]:
                        if key == name:
                            aug["params"][key] = value
                            found = True
                            break

            if not found:
                raise InputError(f"Augmentation {param_def} not found")

    def _get_scores(self):
        import numpy as np
        from britekit.core.trainer import Trainer

        if self.extract:
            # extract a new set of spectrograms to tune spectrogram parameters
            logging.info("Extracting spectrograms")
            self.reextractor.run(quiet=True)
            logging.info("Saving pickle file")
            self.pickler.pickle(quiet=True)

        scores = np.zeros(self.num_runs)
        for i in range(self.num_runs):
            # set different seed each run, but same seed each trial,
            # for variety across runs and stability across trials
            if self.original_seed is None:
                self.cfg.train.seed = 100 + i

            if not self.skip_training:
                Trainer().run()

            scores[i] = self._run_test()

            if self.num_runs > 1:
                logging.info(
                    f"*** Current score={scores[i]:.4f} (trial {self.trial_num + 1}, run {i + 1} of {self.num_runs})"
                )

        return scores

    def _recursive_trials(self, start_index, params):
        """
        Use recursion to exhaustively explore hyperparameter space, and return
        the best combination.
        """

        param_def = self.param_space[start_index]
        values = self._get_values(param_def)

        for value in values:
            params[param_def["name"]] = value
            self._set_value(param_def, value)
            self.trial_metrics[self.trial_num]["params"] = copy.deepcopy(params)

            if start_index == len(self.param_space) - 1:
                scores = self._get_scores()
                score = scores.mean()
                if score > self.best_score:
                    self.best_score = score
                    self.best_scores = scores
                    self.best_params = params.copy()

                logging.info(
                    f"*** Trial score={score:.4f}, params={params}, runs={scores}"
                )
                logging.info(
                    f"*** Best score={self.best_score:.4f}, best params={self.best_params}"
                )

                self.trial_num += 1
                self.trial_metrics[self.trial_num] = {}
            else:
                self._recursive_trials(start_index + 1, params)

    def _random_trials(self):
        """
        Test num_trials random combinations of parameters.
        """

        values = []
        total_combinations = 1
        for i in range(len(self.param_space)):
            values.append(self._get_values(self.param_space[i]))
            total_combinations *= len(values[-1])

        if total_combinations <= self.num_trials:
            # might as well do an exhaustive search
            self._recursive_trials(0, {})
            return

        already_tried = set()
        while self.trial_num < self.num_trials:
            trial = []
            for i in range(len(values)):
                trial.append(random.randint(0, len(values[i]) - 1))

            trial_tuple = tuple(trial)
            if trial_tuple in already_tried:
                continue  # try another one

            already_tried.add(trial_tuple)

            params = {}
            for i in range(len(values)):
                param_def = self.param_space[i]
                params[param_def["name"]] = values[i][trial[i]]
                self._set_value(param_def, values[i][trial[i]])

            self.trial_metrics[self.trial_num]["params"] = params

            scores = self._get_scores()
            score = scores.mean()
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()

            logging.info(f"*** Score={score:.4f}, params={params}, runs={scores}")
            logging.info(
                f"*** Best score={self.best_score:.4f}, best params={self.best_params}"
            )
            self.trial_num += 1
            self.trial_metrics[self.trial_num] = {}

    @staticmethod
    def _find_latest_version_dir(root):
        root = Path(root)
        version_dirs = []
        for d in root.iterdir():
            if d.is_dir() and d.name.startswith("version"):
                m = re.search(r"\d+", d.name)
                if m:
                    version_dirs.append((int(m.group()), d))

        assert version_dirs, "Failed to find training log directory"

        # Sort numerically by the extracted version number
        return max(version_dirs, key=lambda x: x[0])[1].name

    def _run_test(self):
        """
        Run inference with the generated checkpoints and return the selected metric.
        """
        from britekit.core.analyzer import Analyzer
        from britekit.testing.per_segment_tester import PerSegmentTester

        if not self.skip_training:
            train_dir = self._find_latest_version_dir(self.train_log_dir)
            self.cfg.misc.ckpt_folder = str(
                Path(self.train_log_dir) / train_dir / "checkpoints"
            )

        logging.info(f"Using checkpoints in {self.cfg.misc.ckpt_folder}")
        self.cfg.infer.min_score = 0

        # suppress console output during inference and test analysis
        util.set_logging(level=logging.ERROR)

        label_dir = "tuning_labels"
        inference_output_dir = str(Path(self.recording_dir) / label_dir)
        Analyzer().run(self.recording_dir, inference_output_dir)

        with tempfile.TemporaryDirectory() as output_dir:
            tester = PerSegmentTester(
                self.annotation_path,
                self.recording_dir,
                inference_output_dir,
                output_dir,
                self.cfg.infer.min_score,
            )
            tester.initialize()

            pr_stats = tester.get_pr_auc_stats()
            roc_stats = tester.get_roc_auc_stats()

            if "macro_pr_auc" not in self.trial_metrics[self.trial_num]:
                self.trial_metrics[self.trial_num]["macro_pr_auc"] = []
                self.trial_metrics[self.trial_num]["micro_pr_auc"] = []
                self.trial_metrics[self.trial_num]["macro_roc_auc"] = []
                self.trial_metrics[self.trial_num]["micro_roc_auc"] = []

            self.trial_metrics[self.trial_num]["macro_pr_auc"].append(
                pr_stats["macro_pr_auc"]
            )
            self.trial_metrics[self.trial_num]["micro_pr_auc"].append(
                pr_stats["micro_pr_auc_trained"]
            )
            self.trial_metrics[self.trial_num]["macro_roc_auc"].append(
                roc_stats["macro_roc_auc"]
            )
            self.trial_metrics[self.trial_num]["micro_roc_auc"].append(
                roc_stats["micro_roc_auc_trained"]
            )

            if "_pr" in self.metric:
                score = pr_stats[self.metric]
            else:
                score = roc_stats[self.metric]

        util.set_logging()  # restore console output
        return score

    def _write_reports(self):
        # create and write metrics-summary.csv
        import numpy as np
        import pandas as pd

        trials = []
        params = []
        macro_pr_mean = []
        micro_pr_mean = []
        macro_roc_mean = []
        micro_roc_mean = []
        macro_pr_stdev = []
        micro_pr_stdev = []
        macro_roc_stdev = []
        micro_roc_stdev = []
        for trial_num in range(self.trial_num):
            m = self.trial_metrics[trial_num]
            trials.append(trial_num + 1)
            params.append(m["params"])
            macro_pr_mean.append(np.array(m["macro_pr_auc"]).mean())
            micro_pr_mean.append(np.array(m["micro_pr_auc"]).mean())
            macro_roc_mean.append(np.array(m["macro_roc_auc"]).mean())
            micro_roc_mean.append(np.array(m["micro_roc_auc"]).mean())

            macro_pr_stdev.append(np.array(m["macro_pr_auc"]).std())
            micro_pr_stdev.append(np.array(m["micro_pr_auc"]).std())
            macro_roc_stdev.append(np.array(m["macro_roc_auc"]).std())
            micro_roc_stdev.append(np.array(m["micro_roc_auc"]).std())

        df = pd.DataFrame()
        df["Trial #"] = trials
        df["Params"] = params
        df["Macro PR-AUC (mean)"] = macro_pr_mean
        df["Macro PR-AUC (stdev)"] = macro_pr_stdev
        df["Micro PR-AUC (mean)"] = micro_pr_mean
        df["Micro PR-AUC (stdev)"] = micro_pr_stdev
        df["Macro ROC-AUC (mean)"] = macro_roc_mean
        df["Macro ROC-AUC (stdev)"] = macro_roc_stdev
        df["Micro ROC-AUC (mean)"] = micro_roc_mean
        df["Micro ROC-AUC (stdev)"] = micro_roc_stdev

        csv_path = str(Path(self.output_dir) / "metrics-summary.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")

        # create and write metrics-details.csv
        trials = []
        runs = []
        macro_pr = []
        micro_pr = []
        macro_roc = []
        micro_roc = []
        for trial_num in range(self.trial_num):
            m = self.trial_metrics[trial_num]
            for run_num in range(self.num_runs):
                trials.append(trial_num + 1)
                runs.append(run_num + 1)

                macro_pr.append(m["macro_pr_auc"][run_num])
                micro_pr.append(m["micro_pr_auc"][run_num])
                macro_roc.append(m["macro_roc_auc"][run_num])
                micro_roc.append(m["micro_roc_auc"][run_num])

        df = pd.DataFrame()
        df["Trial #"] = trials
        df["Run #"] = runs
        df["Macro PR-AUC"] = macro_pr
        df["Micro PR-AUC"] = micro_pr
        df["Macro ROC-AUC"] = macro_roc
        df["Micro ROC-AUC"] = micro_roc

        csv_path = str(Path(self.output_dir) / "metrics-details.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")

    def run(self):
        """
        Initiate the search and return the best score and best hyperparameter values.
        A "trial" is a set of parameter values and a "run" is a training/inference run.
        There may be multiple runs per trial, since results per run are non-deterministic.
        """
        import numpy as np
        from britekit.training_db.training_db import TrainingDatabase

        self.best_score = float("-inf")
        self.best_params = None
        self.trial_num = 0
        self.trial_metrics = {}  # metrics per trial
        self.trial_metrics[0] = {}
        np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

        if self.param_space is None:
            # just loop with the base config
            self.trial_metrics[0]["params"] = ""
            scores = self._get_scores()
            logging.info(f"*** Scores = {scores}")
            logging.info(
                f"*** Mean = {scores.mean():.4f}, Std Dev = {scores.std():.4f} "
            )
            self.trial_num = 1  # so write_reports doesn't skip trial 0
        elif self.num_trials == 0:
            # num_trials = 0 means do exhaustive search
            self._recursive_trials(0, {})
        else:
            self._random_trials()

        self._write_reports()

        # delete the temporary pickle file and spec_group, if needed
        if self.extract:
            self.pickle_file.close()

            with TrainingDatabase(self.cfg.train.train_db) as db:
                results = db.get_specgroup({"Name": self.spec_group})
                if results:
                    db.delete_specgroup({"ID": results[0].id})

        return self.best_score, self.best_params
