# Defer some imports to improve initialization performance.
from copy import deepcopy
import importlib.util
import logging
import math
import os
from typing import Sequence, Optional, List

import numpy as np

from britekit.core.config_loader import get_config
from britekit.core.exceptions import InferenceError
from britekit.core import util


class Label:
    def __init__(self, score: float, start_time: float, end_time: float) -> None:
        self.score = score
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self) -> str:
        return f"score={self.score:.3f}, start={self.start_time:.2f}, end={self.end_time:.2f}"


class Predictor:
    """
    Given a recording and a model or ensemble of models, provide methods to return scores in several formats.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the Predictor with a model or ensemble of models.

        Args:
        - model_path (str): Path to a checkpoint (.ckpt) or ONNX (.onnx) file,
            or a directory containing multiple checkpoint/ONNX files for an ensemble.
        - device (str, optional): Device to use for inference ('cuda', 'cpu', or 'mps').
            If None, automatically selects the best available device.
        """
        from britekit.core.audio import Audio

        self.cfg = get_config()
        self.audio = Audio()
        self.class_names: Optional[List[str]] = None
        self.class_codes: Optional[List[str]] = None
        self.class_alt_names: Optional[List[str]] = None
        self.class_alt_codes: Optional[List[str]] = None

        ignore_file = self.cfg.misc.ignore_file
        self.ignore_classes = set()  # Initialize empty set by default
        if ignore_file:
            if not os.path.exists(ignore_file):
                raise InferenceError(f'Ignore file "{ignore_file}" not found.')

            self.ignore_classes = set(util.get_file_lines(ignore_file))

        self.device = device or util.get_device()
        if self.device == "cpu" and importlib.util.find_spec("openvino") is not None:
            import openvino as ov

            self.ov = ov
        else:
            self.ov = None

        self._load_models(model_path)

    def get_raw_scores(self, recording_path: str, start_seconds: float = 0):
        """
        Get scores in array format from the loaded models for the given recording.

        Args:
        - recording_path (str): Path to the audio recording file.
        - start_seconds (float): Where to start processing the recording, in seconds from the start.

        Returns:
            tuple: A tuple containing:
                - avg_score (np.ndarray): Average scores across all models in the ensemble.
                  Shape is (num_spectrograms, num_classes).
                - avg_frame_map (np.ndarray, optional): Average frame-level scores if using SED models.
                  Shape is (num_frames, num_classes). None if not using SED models.
                - start_times (list[float]): Start time in seconds for each spectrogram.
        """
        import numpy as np

        if not os.path.exists(recording_path):
            raise InferenceError(f'Recording "{recording_path}" not found')

        if not os.path.isfile(recording_path):
            raise InferenceError(f'Recording "{recording_path}" is not a file')

        self.audio.load(recording_path)

        # Validate audio duration
        audio_duration = self.audio.seconds()
        if audio_duration <= 0:
            raise InferenceError(f"Invalid audio duration: {audio_duration} seconds")

        increment = max(0.5, self.cfg.audio.spec_duration - self.cfg.infer.overlap)
        end_offset = max(increment, audio_duration - increment)
        start_times = util.get_range(start_seconds, end_offset, increment)
        specs, _ = self.audio.get_spectrograms(start_times)
        if specs is None or len(specs) == 0:
            return None, None, []

        specs = specs**self.cfg.infer.audio_power
        specs = specs.unsqueeze(1)  # (N,H,W) -> (N,1,H,W)
        frame_maps = []
        if self.ov:
            scores = self._get_openvino_scores(specs.cpu().numpy())
        else:
            scores = []
            for model in self.models:
                segment_scores, frame_scores = model.predict(specs, self.device)
                scores.append(segment_scores.cpu().detach().numpy())

                if frame_scores is not None:
                    # SED model, so combine all frame scores into one array
                    frame_map = self.to_global_frames(
                        frame_scores,
                        start_times,
                        self.audio.seconds(),
                    )
                    frame_maps.append(frame_map.cpu().detach().numpy())

        # return the average score across models in the ensemble,
        # and the corresponding start_times (spectrogram start times) in the recording
        if not scores:
            raise InferenceError("No scores generated from models")

        avg_score = np.mean(scores, axis=0)
        if len(frame_maps) == 0:
            avg_frame_map = None
        else:
            avg_frame_map = np.mean(frame_maps, axis=0)

        return avg_score, avg_frame_map, start_times

    def get_segment_labels(
        self, scores, start_times: list[float]
    ) -> dict[str, list[Label]]:
        """
        Given an array of raw segment-level scores, return dict of labels.

        Args:
        - scores (np.ndarray): Array of scores of shape (num_spectrograms, num_species).
        - start_times (list[float]): Start time in seconds for each spectrogram.

        Returns:
            dict[str, list]: Dictionary mapping species names to lists of Label objects.
                Each Label contains (score, start_time, end_time) for detected segments.
        """
        assert self.class_names is not None

        labels: dict[str, list] = {}  # name -> [(score, start_time, end_time)]
        if scores is None or len(scores) == 0:
            return labels

        # Validate input dimensions
        if len(scores.shape) != 2:
            raise InferenceError(f"Scores must be 2D array, got shape {scores.shape}")

        if scores.shape[1] != len(self.class_names):
            raise InferenceError(
                f"Number of classes in scores ({scores.shape[1]}) must match number of class names ({len(self.class_names)})"
            )

        names = self._get_names()

        # ensure labels are sorted by name/code before start_time,
        # which is useful when inspecting label files during testing
        num_specs, num_classes = scores.shape
        for i in range(num_specs):
            for j in range(num_classes):
                if self.class_names[j] in self.ignore_classes:
                    continue

                if names[j] not in labels:
                    labels[names[j]] = []

                if scores[i][j] >= self.cfg.infer.min_score:
                    start_time = start_times[i]
                    end_time = start_times[i] + self.cfg.audio.spec_duration
                    score = scores[i][j]
                    labels[names[j]].append(Label(score, start_time, end_time))

        return labels

    def get_frame_labels(self, frame_map) -> dict[str, list[Label]]:
        """
        Given a frame map, return dict of labels.

        Args:
        - frame_map (np.ndarray): Array of scores of shape (num_frames, num_species).

        Returns:
            dict[str, list]: Dictionary mapping species names to lists of Label objects.
                Each Label contains (score, start_time, end_time) for detected segments.
                Labels are either variable-duration (if segment_len is None) or
                fixed-duration based on cfg.infer.segment_len.
        """
        import numpy as np

        assert self.class_names is not None

        # Validate input dimensions
        if len(frame_map.shape) != 2:
            raise InferenceError(
                f"Frame map must be 2D array, got shape {frame_map.shape}"
            )

        if frame_map.shape[1] != len(self.class_names):
            raise InferenceError(
                f"Number of classes in frame_map ({frame_map.shape[1]}) must match number of class names ({len(self.class_names)})"
            )

        names = self._get_names()

        num_frames, num_classes = frame_map.shape
        frames_per_second = self.cfg.train.sed_fps
        labels: dict[str, list] = {}  # name -> [(score, start_time, end_time)]

        if self.cfg.infer.segment_len is None:
            # variable-duration labels
            for i in range(num_classes):
                if self.class_names[i] in self.ignore_classes:
                    continue

                labels[names[i]] = []
                curr_label = None
                curr_seg_len: float = 0.0
                # process one frame at a time
                for j in range(num_frames):
                    if frame_map[j, i] >= self.cfg.infer.min_score:
                        score = round(frame_map[j, i], 3)
                        frame_time = round(j / frames_per_second, 3)
                        if curr_label is None:
                            # start new label
                            curr_label = Label(score, frame_time, frame_time)
                        else:
                            # update current label
                            curr_label.score = max(score, curr_label.score)
                            curr_label.end_time = frame_time + 1 / frames_per_second
                    elif curr_label is not None:
                        # end current label
                        labels[names[i]].append(curr_label)
                        curr_label = None
                        curr_seg_len = 0.0

                    curr_seg_len += 1 / frames_per_second

                if curr_label is not None:
                    labels[names[i]].append(curr_label)
        else:
            # fixed-duration labels
            num_frame_seconds = num_frames / frames_per_second
            num_frame_segments = math.ceil(
                num_frame_seconds / self.cfg.infer.segment_len
            )
            frames_per_segment = frames_per_second * self.cfg.infer.segment_len
            for i in range(num_classes):
                if self.class_names[i] in self.ignore_classes:
                    continue

                labels[names[i]] = []
                # process one segment at a time
                for j in range(num_frame_segments):
                    start_idx = int(j * frames_per_segment)
                    end_idx = int((j + 1) * frames_per_segment)
                    score = np.max(frame_map[start_idx:end_idx, i])
                    if score >= self.cfg.infer.min_score:
                        score = round(score, 3)
                        start_time = j * self.cfg.infer.segment_len
                        end_time = (j + 1) * self.cfg.infer.segment_len
                        labels[names[i]].append(Label(score, start_time, end_time))

        return labels

    def get_dataframe(
        self,
        score_array,
        frame_map,
        start_times: list[float],
        recording_name: str,
    ):
        """
        Given an array of raw scores, return as a pandas dataframe.

        Args:
        - score_array (np.ndarray): Array of scores of shape (num_spectrograms, num_species).
        - frame_map (np.ndarray, optional): Frame-level scores of shape (num_frames, num_species).
            If provided, uses frame-level labels; otherwise uses segment-level labels.
        - start_times (list[float]): Start time in seconds for each spectrogram.
        - recording_name (str): Name of the recording for the dataframe.

        Returns:
            pd.DataFrame: DataFrame with columns ['recording', 'name', 'start_time', 'end_time', 'score']
                containing all detected species segments.
        """
        import pandas as pd

        if frame_map is None:
            labels = self.get_segment_labels(score_array, start_times)
        else:
            labels = self.get_frame_labels(frame_map)

        names = []
        recordings = []
        score_list = []
        start_times = []
        end_times = []
        for name in sorted(labels):
            for label in labels[name]:
                names.append(name)
                recordings.append(recording_name)
                score_list.append(label.score)
                start_times.append(label.start_time)
                end_times.append(label.end_time)

        df = pd.DataFrame()
        df["recording"] = recordings
        df["name"] = names
        df["start_time"] = start_times
        df["end_time"] = end_times
        df["score"] = score_list
        return df

    def log_scores(self, scores):
        """
        Given an array of raw segment-level scores, log them by descending score.

        Args:
        - scores (np.ndarray): Array of scores of shape (num_spectrograms, num_species).
        """
        assert self.class_names is not None

        labels: dict[str, list] = {}  # name -> [(score, start_time, end_time)]
        if scores is None or len(scores) == 0:
            return labels

        names = self._get_names()

        # ensure labels are sorted by name/code before start_time,
        # which is useful when inspecting label files during testing
        num_classes = scores.shape[1]
        scores = deepcopy(scores[0])
        for i in range(min(num_classes, 10)):
            j = np.argmax(scores)
            logging.info(f"{names[j]}: {scores[j]:.4f}")
            scores[j] = 0

    def save_audacity_labels(
        self,
        scores,
        frame_map,
        start_times: list[float],
        file_path: str,
    ) -> None:
        """
        Given an array of raw scores, convert to Audacity labels and save in the given file.

        Args:
        - scores (np.ndarray): Segment-level scores of shape (num_spectrograms, num_species).
        - frame_map (np.ndarray, optional): Frame-level scores of shape (num_frames, num_species).
            If provided, uses frame-level labels; otherwise uses segment-level labels.
        - start_times (list[float]): Start time in seconds for each spectrogram.
        - file_path (str): Output path for the Audacity label file.

        Returns:
            None: Writes the labels directly to the specified file.
        """

        if frame_map is None:
            labels = self.get_segment_labels(scores, start_times)
        else:
            labels = self.get_frame_labels(frame_map)

        try:
            with open(file_path, "w") as out_file:
                for name in sorted(labels):
                    for label in labels[name]:
                        text = f"{label.start_time:.2f}\t{label.end_time:.2f}\t{name};{label.score:.3f}\n"
                        out_file.write(text)
        except (IOError, OSError) as e:
            raise InferenceError(
                f"Failed to write Audacity labels to {file_path}: {str(e)}"
            )

    def to_global_frames(
        self,
        frame_scores,
        offsets_sec: Sequence[float],
        recording_duration_sec: float,
    ):
        """
        Map overlapping per-spectrogram frame scores onto a global frame grid.
        Use mean rather than max or weighted values.

        Args:
        - frame_scores: (num_specs, num_classes, T_spec) scores in [0, 1].
        - offsets_sec: start time (s) for each spectrogram within the recording.
        - recording_duration_sec: total recording length in seconds.

        Returns:
            global_frames: (num_classes, T_global) tensor of scores in [0, 1].
        """
        import torch

        with torch.no_grad():
            assert (
                frame_scores.dim() == 3
            ), "frame_scores must be (num_specs, num_classes, T_spec)"
            num_specs, num_classes, T_spec = frame_scores.shape
            device = frame_scores.device
            fps = self.cfg.train.sed_fps

            # Validate fps
            if fps <= 0:
                raise InferenceError(f"Invalid fps value: {fps}")

            # Global grid length (frames)
            T_global: int = int(round(fps * recording_duration_sec))

            # Map spectrogram offsets (seconds) to global frame indices
            starts = torch.tensor(
                [int(round(o * fps)) for o in offsets_sec],
                device=device,
                dtype=torch.int64,
            )

            # Prepare accumulation buffers
            global_scores = torch.zeros((T_global, num_classes), device=device)
            weights = torch.zeros((T_global,), device=device)

            # Accumulate
            assert global_scores is not None
            for k in range(num_specs):
                start: int = int(starts[k].item())
                # Skip chunks entirely outside the global range
                if start >= T_global or start + T_spec <= 0:
                    continue

                g0: int = int(max(0, start))
                g1: int = int(min(T_global, start + T_spec))
                t0: int = int(g0 - start)
                t1: int = int(t0 + (g1 - g0))

                chunk = frame_scores[k, :, t0:t1].T  # shape (local_T, num_classes)
                assert weights is not None
                global_scores[g0:g1, :] += chunk
                weights[g0:g1] += 1.0

            # Finalize
            assert weights is not None
            denom = torch.clamp(weights, min=1e-12).unsqueeze(1)  # (T_global, 1)
            return global_scores / denom

    # =============================================================================
    # Private Helper Methods
    # =============================================================================

    def _load_models(self, model_path: str) -> None:
        """Given a checkpoint path or directory, load and return a list of models"""
        self.models = []
        if not os.path.exists(model_path):
            raise InferenceError(f'Model path "{model_path}" not found')

        if os.path.isfile(model_path):
            if self.ov:
                if model_path.endswith(".onnx"):
                    self.models.append(self._load_model(model_path))
                else:
                    raise InferenceError(
                        f'"{model_path}" does not end with .onnx when using openvino'
                    )
            elif model_path.endswith(".ckpt"):
                self.models.append(self._load_model(model_path).to(self.device))
            else:
                raise InferenceError(f'"{model_path}" does not end with .ckpt')
        else:
            for file_path in sorted(os.listdir(model_path)):
                full_path = os.path.join(model_path, file_path)
                if not os.path.isfile(full_path):
                    continue

                # no exceptions needed in the directory case, since we can ignore irrelevant files
                if self.ov:
                    if full_path.endswith(".onnx"):
                        self.models.append(self._load_model(full_path))
                elif full_path.endswith(".ckpt"):
                    self.models.append(self._load_model(full_path).to(self.device))

        if not self.models:
            raise InferenceError(f'No models loaded from "{model_path}"')

    def _load_model(self, model_path: str):
        """Given a checkpoint path, load and return a single model"""
        from britekit.models import model_loader

        if model_path.endswith(".ckpt"):
            try:
                model = model_loader.load_from_checkpoint(model_path).eval()
                if self.class_names is None:
                    self.class_names = model.train_class_names
                    self.class_codes = model.train_class_codes
                    self.class_alt_names = model.train_class_alt_names
                    self.class_alt_codes = model.train_class_alt_codes

                    # set defaults for missing alt names and codes
                    for i, name in enumerate(self.class_alt_names):
                        if not name:
                            self.class_alt_names[i] = self.class_names[i]

                    for i, code in enumerate(self.class_alt_codes):
                        if not code:
                            self.class_alt_codes[i] = self.class_codes[i]
            except Exception as e:
                raise InferenceError(
                    f"Failed to load model from {model_path}: {str(e)}"
                )

        elif model_path.endswith(".onnx"):
            try:
                core = self.ov.Core()
                model_onnx = core.read_model(model=model_path)
                model = core.compile_model(model=model_onnx, device_name="CPU")
            except Exception as e:
                raise InferenceError(
                    f"Failed to load ONNX model from {model_path}: {str(e)}"
                )
        else:
            raise InferenceError(f"Unsupported model format: {model_path}")

        return model

    def _get_openvino_scores(self, specs):
        import numpy as np
        import torch

        scores = []
        block_size = self.cfg.infer.openvino_block_size
        num_blocks = (specs.shape[0] + block_size - 1) // block_size

        for model in self.models:
            try:
                output_layer = model.output(0)
            except Exception as e:
                raise InferenceError(f"Failed to get model output layer: {str(e)}")
            model_scores = []

            for i in range(num_blocks):
                # slice the input into blocks of size block_size
                start_idx = i * block_size
                end_idx = min((i + 1) * block_size, specs.shape[0])
                block = specs[start_idx:end_idx]

                # pad the block with zeros if it's smaller than block_size
                if block.shape[0] < block_size:
                    pad_shape = (block_size - block.shape[0], *block.shape[1:])
                    padding = np.zeros(pad_shape, dtype=block.dtype)
                    block = np.concatenate((block, padding), axis=0)

                # run inference on the block
                result = model(block)[output_layer]
                result = torch.sigmoid(torch.tensor(result)).cpu().numpy()

                # trim the padded scores to match the original block size
                model_scores.append(result[: end_idx - start_idx])

            # combine scores for the model
            scores.append(np.concatenate(model_scores, axis=0))

        return scores

    def _get_names(self) -> list[str]:
        """Return list of names as specifed by cfg.infer.label_field"""
        if self.class_names is None or self.class_codes is None:
            raise InferenceError("Class names and codes not defined")

        names = self.class_names
        if self.cfg.infer.label_field == "codes":
            if None not in self.class_codes:
                names = self.class_codes
        elif self.cfg.infer.label_field == "alt_names":
            if self.class_alt_names is None:
                raise InferenceError("Alt names not available")
            if None not in self.class_alt_names:
                names = self.class_alt_names
        elif self.cfg.infer.label_field == "alt_codes":
            if self.class_alt_codes is None:
                raise InferenceError("Alt codes not available")
            if None not in self.class_alt_codes:
                names = self.class_alt_codes
        elif self.cfg.infer.label_field != "names":
            logging.error(
                f'Invalid label_field option ("{self.cfg.infer.label_field}"). Defaulting to class names.'
            )

        return names
