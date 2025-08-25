import os
from pathlib import Path
import random
import re
import tempfile

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
        recording_dir,
        annotation_path,
        train_log_dir,
        trained_species_codes,
        metric,
        param_space,
        num_trials: int = 0,
        num_runs: int = 1,
    ):
        self.cfg, self.fn_cfg = get_config()
        self.original_seed = self.cfg.train.seed
        self.recording_dir = recording_dir
        self.annotation_path = annotation_path
        self.train_log_dir = train_log_dir
        self.trained_species_codes = trained_species_codes
        self.metric = metric
        self.param_space = param_space
        self.num_trials = num_trials
        self.num_runs = num_runs

        if self.metric not in {
            "macro_map",
            "micro_map_annotated",
            "micro_map_trained",
            "combined_map_annotated",
            "combined_map_trained",
            "macro_roc",
            "micro_roc_annotated",
            "micro_roc_trained",
            "combined_roc_annotated",
            "combined_roc_trained",
        }:
            raise InputError(f"Invalid metric: {self.metric}")

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
                    np.set_printoptions(precision=4, suppress=True)
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

        output_dir = tempfile.gettempdir()
        tester = PerSegmentTester(
            self.annotation_path,
            self.recording_dir,
            inference_output_dir,
            output_dir,
            self.cfg.infer.min_score,
            self.trained_species_codes,
        )
        tester.initialize()

        if "_map" in self.metric:
            stats = tester.get_map_stats()
        else:
            stats = tester.get_roc_stats()

        self.fn_cfg.echo = echo  # restore console output
        return stats[self.metric]

    def run(self):
        """
        Initiate the search and return the best score and best hyperparameter values.
        """
        self.best_score = float("-inf")
        self.best_params = None

        if self.num_trials == 0:
            # num_trials = 0 means do exhaustive search
            self._recursive_trials(0, {})
        else:
            self._random_trials()

        return self.best_score, self.best_params
