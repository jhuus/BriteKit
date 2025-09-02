import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Optional

import numpy as np

from britekit.core.analyzer import Analyzer
from britekit.core.config_loader import get_config
from britekit.core.exceptions import InputError
from britekit.core.trainer import Trainer
from britekit.core import util
from britekit.testing.per_segment_tester import PerSegmentTester


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
        annotation_path: str,
        train_log_dir: str,
        metric: str,
        param_space: Optional[Any],
        num_trials: int = 0,
        num_runs: int = 1,
    ):
        self.cfg, self.fn_cfg = get_config()
        self.original_seed = self.cfg.train.seed
        self.recording_dir = recording_dir
        self.annotation_path = annotation_path
        self.train_log_dir = train_log_dir
        self.param_space = param_space
        self.num_trials = num_trials
        self.num_runs = num_runs

        # lists to track and report all scores
        self.macro_pr_scores = []
        self.micro_pr_scores = []
        self.macro_roc_scores = []
        self.micro_roc_scores = []

        # map short metric name to full name
        metric_dict = {
            "macro_pr": "macro_pr_auc",
            "micro_pr": "micro_pr_auc_trained",
            "macro_roc": "macro_roc_auc",
            "micro_roc": "micro_roc_auc_trained",
        }

        if metric not in metric_dict:
            raise InputError(f"Invalid metric: {self.metric}")

        self.metric = metric_dict[metric]
        util.echo(f"Using metric {metric} (full name = {self.metric})")

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

        util.echo(f"*** set {name}={value}")
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
        scores = np.zeros(self.num_runs)
        for i in range(self.num_runs):
            # set different seed each run, but same seed each trial,
            # for variety across runs and stability across trials
            if self.original_seed is None:
                self.cfg.train.seed = 100 + i
            Trainer().run()
            scores[i] = self._run_test()

            if self.num_runs > 1:
                util.echo(f"*** current score={scores[i]:.4f}")

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

            if start_index == len(self.param_space) - 1:
                scores = self._get_scores()
                score = scores.mean()
                if score > self.best_score:
                    self.best_score = score
                    self.best_scores = scores
                    self.best_params = params.copy()

                util.echo(f"*** score={score:.4f}, params={params}")
                util.echo(
                    f"*** best score={self.best_score:.4f}, best params={self.best_params}"
                )

                if self.num_runs > 1:
                    util.echo(f"*** scores={scores}, stdev={np.std(scores):.4f}")
                    util.echo(
                        f"*** best scores={self.best_scores}, stdev={np.std(self.best_scores):.4f}"
                    )
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
        trial_num = 0
        while trial_num < self.num_trials:
            trial = []
            for i in range(len(values)):
                trial.append(random.randint(0, len(values[i]) - 1))

            trial_tuple = tuple(trial)
            if trial_tuple in already_tried:
                continue  # try another one

            already_tried.add(trial_tuple)
            trial_num += 1

            params = {}
            for i in range(len(values)):
                param_def = self.param_space[i]
                params[param_def["name"]] = values[i][trial[i]]
                self._set_value(param_def, values[i][trial[i]])

            scores = self._get_scores()
            score = scores.mean()
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()

            util.echo(f"*** score={score:.4f}, params={params}")
            util.echo(
                f"*** best score={self.best_score:.4f}, best params={self.best_params}"
            )

    def _run_test(self):
        """
        Run inference with the generated checkpoints and return the selected metric.
        """

        train_dir = sorted(os.listdir(self.train_log_dir), key=natural_key)[-1]
        self.cfg.misc.ckpt_folder = str(
            Path(self.train_log_dir) / train_dir / "checkpoints"
        )
        self.cfg.infer.min_score = 0

        echo = self.fn_cfg.echo
        self.fn_cfg.echo = (
            None  # suppress console output during inference and test analysis
        )
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

            self.macro_pr_scores.append(pr_stats['macro_pr_auc'])
            self.micro_pr_scores.append(pr_stats['micro_pr_auc_trained'])
            self.macro_roc_scores.append(roc_stats['macro_roc_auc'])
            self.micro_roc_scores.append(roc_stats['micro_roc_auc_trained'])

            if "_pr" in self.metric:
                score = pr_stats[self.metric]
            else:
                score = roc_stats[self.metric]

        self.fn_cfg.echo = echo  # restore console output
        return score

    def run(self):
        """
        Initiate the search and return the best score and best hyperparameter values.
        """
        self.best_score = float("-inf")
        self.best_params = None
        np.set_printoptions(precision=4, suppress=True)

        if self.param_space is None:
            # just loop with the base config
            scores = self._get_scores()
            util.echo(f"*** Scores = {scores}")
            util.echo(f"*** Average = {scores.mean():.4f}, Std Dev = {scores.std():.4f} ")
        elif self.num_trials == 0:
            # num_trials = 0 means do exhaustive search
            self._recursive_trials(0, {})
        else:
            self._random_trials()

        # Print all the stats
        macro_pr_scores = np.array(self.macro_pr_scores)
        micro_pr_scores = np.array(self.micro_pr_scores)
        macro_roc_scores = np.array(self.macro_roc_scores)
        micro_roc_scores = np.array(self.micro_roc_scores)

        util.echo()
        util.echo(f"Macro PR-AUC scores = {macro_pr_scores}")
        util.echo(f"Macro PR-AUC mean = {macro_pr_scores.mean():.4f}, stdev = {macro_pr_scores.std():.4f}")
        util.echo(f"Micro PR-AUC scores = {micro_pr_scores}")
        util.echo(f"Micro PR-AUC mean = {micro_pr_scores.mean():.4f}, stdev = {micro_pr_scores.std():.4f}")
        util.echo(f"Macro ROC-AUC scores = {macro_roc_scores}")
        util.echo(f"Macro ROC-AUC mean = {macro_roc_scores.mean():.4f}, stdev = {macro_roc_scores.std():.4f}")
        util.echo(f"Micro ROC-AUC scores = {micro_roc_scores}")
        util.echo(f"Micro ROC-AUC mean = {micro_roc_scores.mean():.4f}, stdev = {micro_roc_scores.std():.4f}")

        return self.best_score, self.best_params
