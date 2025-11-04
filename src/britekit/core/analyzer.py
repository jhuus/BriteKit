# Defer some imports to improve initialization performance.
import logging
import os
from pathlib import Path
import threading
from typing import Dict, Any

from britekit.core.config_loader import get_config
from britekit.core.exceptions import InferenceError
from britekit.core import util


class Analyzer:
    """
    Basic inference logic using Predictor class, with multi-threading and multi-recording support.
    """

    def __init__(self):
        self.cfg = get_config()
        self.dataframes = []

    def _save_manifest(self, output_path: str, predictor):
        """
        Save a text file summarizing the inference configuration.
        """
        import yaml
        from britekit.models.base_model import BaseModel

        # Add class list
        model: BaseModel = predictor.models[0]
        names = model.train_class_names
        codes = model.train_class_codes

        info: Dict[str, Any] = {}
        classes = []
        for i, name in enumerate(names):
            classes.append({"name": name, "code": codes[i]})
        info["classes"] = classes

        # Add current inference config
        info["audio"] = util.cfg_to_pure(self.cfg.audio)
        info["inference"] = util.cfg_to_pure(self.cfg.infer)

        # Add config per model
        for i, model in enumerate(predictor.models):
            key = f"model {i + 1}"
            info[key] = {}
            info[key]["identifier"] = model.identifier
            info[key]["training_date"] = model.training_date
            info[key]["audio"] = model.training_cfg["audio"]
            info[key]["train"] = model.training_cfg["train"]

        # Write the manifest
        info_str = yaml.dump(info, sort_keys=False)
        info_str = "# Summary of inference run in YAML format\n" + info_str
        with open(Path(output_path) / "manifest.yaml", "w") as out_file:
            out_file.write(info_str)

    def _process_recordings(
        self, recording_paths, output_path, rtype, start_seconds, thread_num, debug_mode=False
    ):
        """
        This runs on its own thread and processes all recordings in the given list.

        Args:
        - recording_paths (list): Individual recording paths.
        - output_path (str): Where to write the output.
        - rtype (str): Output format: "audacity", "csv" or "both".
        - start_seconds (float): Where to start processing each recording, in seconds from start.
        """
        from britekit.core.predictor import Predictor

        predictor = Predictor(self.cfg.misc.ckpt_folder)
        for recording_path in recording_paths:
            logging.info(f"[Thread {thread_num}] Processing {recording_path}")
            scores, frame_map, offsets = predictor.get_raw_scores(
                recording_path, start_seconds
            )
            if debug_mode:
                predictor.log_scores(scores)  # log the scores for debugging

            recording_name = Path(recording_path).stem
            if rtype in {"audacity", "both"}:
                file_path = str(Path(output_path) / f"{recording_name}_scores.txt")
                predictor.save_audacity_labels(scores, frame_map, offsets, file_path)

            if rtype in {"csv", "both"}:
                dataframe = predictor.get_dataframe(
                    scores, frame_map, offsets, recording_name
                )
                self.dataframes.append(dataframe)

            if debug_mode:
                break

        if thread_num == 1:
            self._save_manifest(output_path, predictor)

    @staticmethod
    def _split_list(input_list, n):
        """
        Split the input list into `n` lists based on index modulo `n`.

        Args:
        - input_list (list): The input list to split.
        - n (int): Number of resulting groups.

        Returns:
            List[List]: A list of `n` lists, where each sublist contains elements
                        whose indices mod n are equal.
        """
        result = [[] for _ in range(n)]
        for i, item in enumerate(input_list):
            result[i % n].append(item)
        return result

    def run(
        self,
        input_path: str,
        output_path: str,
        rtype: str = "audacity",
        start_seconds: float = 0,
        debug_mode: bool = False
    ):
        """
        Run inference.

        Args:
        - input_path (str): Recording or directory containing recordings.
        - output_path (str): Output directory.
        - rtype (str): Output format: "audacity", "csv" or "both".
        - start_seconds (float): Where to start processing each recording, in seconds.
        - debug_mode (bool): If true, log scores for the first spectrogram, then stop.
        For example, '71' and '1:11' have the same meaning, and cause the first 71 seconds to be ignored. Default = 0.
        """
        import pandas as pd

        if os.path.isfile(input_path):
            recording_paths = [input_path]
        else:
            recording_paths = util.get_audio_files(input_path)
            if len(recording_paths) == 0:
                raise InferenceError(f'No audio recordings found in "{input_path}"')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.dataframes = []
        num_threads = min(self.cfg.infer.num_threads, len(recording_paths))
        if num_threads == 1:
            self._process_recordings(
                recording_paths, output_path, rtype, start_seconds, 1, debug_mode,
            )
        else:
            recordings_per_thread = self._split_list(recording_paths, num_threads)
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(
                    target=self._process_recordings,
                    args=(
                        recordings_per_thread[i],
                        output_path,
                        rtype,
                        start_seconds,
                        i + 1,
                        debug_mode,
                    ),
                )
                thread.start()
                threads.append(thread)

            for thread in threads:
                # thread exceptions should be handled in caller
                thread.join()

        if rtype in {"csv", "both"}:
            file_path = os.path.join(output_path, "scores.csv")
            combined_df = pd.concat(self.dataframes, ignore_index=True)
            sorted_df = combined_df.sort_values(by=["recording", "name", "start_time"])
            sorted_df.to_csv(file_path, index=False, float_format="%.3f")
