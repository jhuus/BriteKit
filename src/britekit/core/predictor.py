#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
from copy import deepcopy
import importlib.util
import logging
import os
from typing import cast, Any, Dict, List, Optional, Sequence

import numpy as np

from britekit.core.base_config import BaseConfig
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

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        cfg: Optional[BaseConfig] = None,
    ):
        """
        Initialize the Predictor with a model or ensemble of models.

        Args:
        - model_path (str): Path to a checkpoint (.ckpt) or ONNX (.onnx) file,
            or a directory containing multiple checkpoint/ONNX files for an ensemble.
        - device (str, optional): Device to use for inference ('cuda', 'cpu', or 'mps').
            If None, automatically selects the best available device.
        """
        from britekit.core.audio import Audio

        if cfg is None:
            self.cfg = get_config()
        else:
            self.cfg = cfg

        self.audio = Audio(cfg=self.cfg)
        self.class_names: Optional[List[str]] = None
        self.class_codes: Optional[List[str]] = None
        self.class_alt_names: Optional[List[str]] = None
        self.class_alt_codes: Optional[List[str]] = None

        self.last_frame_map = None
        self.labels = None
        self.normalized_specs = None
        self.unnormalized_specs = None

        self.device = device or util.get_device()
        if self.device == "cpu" and importlib.util.find_spec("openvino") is not None:
            import openvino as ov

            self.ov = ov
        else:
            self.ov = None

        self._load_models(model_path)

    def get_embeddings(self, spec_array):
        """
        Given an array of spectrograms, return the average embeddings using the loaded models.

        Args:
        - spec_array: Spectrograms (numpy array).

        Returns:
            Average embeddings (numpy array).
        """
        combined_embeddings = []
        for i, model in enumerate(self.models):
            embeddings = model.get_embeddings(spec_array, self.device)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            combined_embeddings.append(embeddings)
            if i > 0:
                assert (
                    embeddings.shape == combined_embeddings[0].shape
                ), "Mismatched embedding shapes"

        # Stack and average
        combined_embeddings = np.stack(combined_embeddings, axis=0)
        average_embeddings = np.mean(combined_embeddings, axis=0)

        return average_embeddings

    def get_block_scores(self, specs, start_times=None, audio_duration=None):
        """
        Get scores in array format from the loaded models for the given block of spectrograms.

        Args:
        - specs: Spectrograms.
        - start_times: Start time per spectrogram, in seconds from start of recording.
          This is optional and used with SED models only.

        Returns:
            tuple: A tuple containing:
                - avg_score (np.ndarray): Average scores across all models in the ensemble.
                  Shape is (num_spectrograms, num_classes).
                - avg_frame_map (np.ndarray, optional): Average frame-level scores if using SED models.
                  Shape is (num_frames, num_classes). None if not using SED models.
        """
        frame_maps = []
        if self.ov:
            scores = self._get_openvino_scores(specs)
        else:
            scores = []
            for model in self.models:
                segment_scores, frame_scores = model.predict(specs, self.device)

                scores.append(segment_scores)

                if frame_scores is not None and start_times is not None:
                    # SED model, so combine all frame scores into one array
                    if audio_duration is None:
                        audio_duration = self.audio.seconds()

                    frame_map = self.to_global_frames(
                        frame_scores,
                        start_times,
                        audio_duration,
                    )
                    frame_maps.append(frame_map)

        # return the average score across models in the ensemble,
        # and the corresponding start_times (spectrogram start times) in the recording
        if not scores:
            raise InferenceError("No scores generated from models")

        avg_score = np.mean(scores, axis=0)
        if len(frame_maps) == 0:
            avg_frame_map = None
        else:
            avg_frame_map = np.mean(frame_maps, axis=0)

        return avg_score, avg_frame_map

    def get_recording_scores(self, recording_path: str, start_seconds: float = 0):
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
        if not os.path.exists(recording_path):
            raise InferenceError(f'Recording "{recording_path}" not found')

        if not os.path.isfile(recording_path):
            raise InferenceError(f'Recording "{recording_path}" is not a file')

        self.audio.load(recording_path)

        # Validate audio duration
        audio_duration = self.audio.seconds()
        if audio_duration <= 0:
            logging.error(f"Invalid audio duration: {audio_duration} seconds")
            return None, None, []

        start_times = self._get_start_times(audio_duration, start_seconds)
        logging.debug("Predictor::get_recording_scores start_times=%s", start_times)
        specs, self.unnormalized_specs = self.audio.get_spectrograms(start_times)
        self.normalized_specs = specs

        if specs is None or len(specs) == 0:
            return None, None, []

        specs = specs**self.cfg.infer.audio_power
        specs = np.expand_dims(specs, axis=1)  # (N,H,W) -> (N,1,H,W)
        logging.debug("Predictor::get_recording_scores start call to get_block_scores")
        avg_score, avg_frame_map = self.get_block_scores(specs, start_times)
        logging.debug("Predictor::get_recording_scores finish call to get_block_scores")

        return avg_score, avg_frame_map, start_times

    def get_overlapping_scores(
        self, recording_path: str, increment: float, start_seconds: float = 0
    ):
        """
        This is a variant of get_recording_scores that overlaps spectrograms differently,
        and is intended for models with SED classifier heads only. Each model processes
        non-overlapping spectrograms. The first one gets spectrograms starting at offset 0.
        The second one gets spectrograms starting at increment, the third starts at 2*increment,
        etc. Since each model processes non-overlapping spectrograms, this achieves overlapped
        processing with much better performance than get_recording_scores, in which each model
        processes the same overlapping spectrograms.

        Args:
        - recording_path (str): Path to the audio recording file.
        - increment (float): Amount of increment.

        Returns:
            tuple: A tuple containing:
                - avg_score (np.ndarray): Average scores across all models in the ensemble.
                  Shape is (num_spectrograms, num_classes).
                - avg_frame_map (np.ndarray, optional): Average frame-level scores if using SED models.
                  Shape is (num_frames, num_classes). None if not using SED models.
                - start_times (list[float]): Start time in seconds for each spectrogram.
        """
        if not os.path.exists(recording_path):
            raise InferenceError(f'Recording "{recording_path}" not found')

        if not os.path.isfile(recording_path):
            raise InferenceError(f'Recording "{recording_path}" is not a file')

        self.audio.load(recording_path)

        audio_duration = self.audio.seconds()
        if audio_duration <= 0:
            logging.error(f"Invalid audio duration: {audio_duration} seconds")
            return None, None, []

        frame_maps = []
        for i, model in enumerate(self.models):
            # add increment, but never start past the first segment
            curr_start = (start_seconds + i * increment) % self.cfg.audio.spec_duration
            start_times = self._get_start_times(audio_duration, curr_start, overlap=0)
            if len(start_times) > 0 and start_times[0] > start_seconds:
                # extra overlap at the beginning
                start_times = [start_seconds] + start_times

            specs, _ = self.audio.get_spectrograms(start_times)
            if specs is None or len(specs) == 0:
                # maybe recording is too short given the increment
                continue

            specs = specs**self.cfg.infer.audio_power
            specs = np.expand_dims(specs, axis=1)  # (N,H,W) -> (N,1,H,W)
            _, frame_scores = model.predict(specs, self.device)
            if frame_scores is None:
                logging.error(f"No frame scores generated for {recording_path}")
                return None, None, []

            frame_map = self.to_global_frames(
                frame_scores,
                start_times,
                audio_duration,
            )
            frame_maps.append(frame_map)

        if len(frame_maps) == 0:
            return None, None, []

        return np.mean(frame_maps, axis=0)

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
        num_classes = scores.shape[1]
        for i in range(num_classes):
            indexes = np.where(scores[:, i] >= self.cfg.infer.min_score)[0]
            if len(indexes) > 0:
                labels[names[i]] = []
                for index in indexes:
                    start_time = start_times[index]
                    end_time = start_time + self.cfg.audio.spec_duration
                    score = scores[index][i]
                    labels[names[i]].append(Label(score, start_time, end_time))

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
        if frame_map.ndim != 2:
            raise InferenceError(
                f"Frame map must be 2D array, got shape {frame_map.shape}"
            )

        if frame_map.shape[1] != len(self.class_names):
            raise InferenceError(
                f"Number of classes in frame_map ({frame_map.shape[1]}) "
                f"must match number of class names ({len(self.class_names)})"
            )

        # Avoid recomputation if called repeatedly on same object
        if self.last_frame_map is not None and id(self.last_frame_map) == id(frame_map):
            return self.labels

        names = self._get_names()
        num_frames, num_classes = frame_map.shape

        fps = self.cfg.train.sed_fps
        min_score = self.cfg.infer.min_score
        segment_len = self.cfg.infer.segment_len

        labels: dict[str, list[Label]] = {}

        # ------------------------------------------------------------------
        # Variable-duration labels
        # ------------------------------------------------------------------
        if segment_len is None:
            for i, name in enumerate(names):
                col = frame_map[:, i]
                active = col >= min_score

                if not np.any(active):
                    labels[name] = []
                    continue

                # Find contiguous True runs
                diff = np.diff(active.astype(np.int8))
                starts = np.where(diff == 1)[0] + 1
                ends = np.where(diff == -1)[0] + 1

                if active[0]:
                    starts = np.r_[0, starts]
                if active[-1]:
                    ends = np.r_[ends, len(active)]

                class_labels = []
                for s, e in zip(starts, ends):
                    score = np.max(col[s:e])
                    class_labels.append(
                        Label(
                            float(score),
                            s / fps,
                            e / fps,
                        )
                    )

                labels[name] = class_labels

        else:
            # ------------------------------------------------------------------
            # Fixed-duration labels
            # ------------------------------------------------------------------
            frames_per_segment = int(round(fps * segment_len))
            if frames_per_segment <= 0:
                raise InferenceError("frames_per_segment must be > 0")

            num_segments = (num_frames + frames_per_segment - 1) // frames_per_segment

            # Pad frames once so reshape is safe
            pad = num_segments * frames_per_segment - num_frames
            if pad > 0:
                frame_map_padded = np.pad(
                    frame_map,
                    ((0, pad), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
            else:
                frame_map_padded = frame_map

            # Shape: (segments, frames_per_segment, classes)
            seg_view = frame_map_padded.reshape(
                num_segments, frames_per_segment, num_classes
            )

            # Fast reduction over frames to (segments, classes)
            seg_scores = np.max(seg_view, axis=1)

            # Precompute segment times once
            segment_starts = np.arange(num_segments, dtype=np.float32) * segment_len
            segment_ends = segment_starts + segment_len

            min_score = self.cfg.infer.min_score

            for i, name in enumerate(names):
                scores_i = seg_scores[:, i]
                js = np.flatnonzero(scores_i >= min_score)

                if js.size == 0:
                    labels[name] = []
                    continue

                labels[name] = [
                    Label(
                        float(scores_i[j]),
                        float(segment_starts[j]),
                        float(segment_ends[j]),
                    )
                    for j in js
                ]

        self.last_frame_map = cast(Any, frame_map)
        self.labels = cast(Any, labels)
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

    def get_specs(self):
        return self.normalized_specs, self.unnormalized_specs

    def show_scores(self, scores, frame_map):
        """
        Given an array of raw segment-level scores, log them by descending score.
        If frame_map is specified, use that. Otherwise use scores.

        Args:
        - scores (np.ndarray): Array of scores of shape (num_spectrograms, num_species).
        - frame_map (np.ndarray, optional): Array of scores of shape (num_frames, num_species).
        """
        assert self.class_names is not None

        labels: dict[str, list] = {}  # name -> [(score, start_time, end_time)]
        names = self._get_names()

        # if there is a frame_map, convert it to scores
        if frame_map is not None:
            num_frames = int(self.cfg.train.sed_fps * self.cfg.audio.spec_duration)
            scores = frame_map[:num_frames, :].max(axis=0)
            scores = scores[None, :]  # make the shape (1, num_classes)

        if scores is None or len(scores) == 0:
            return labels

        # ensure labels are sorted by name/code before start_time,
        # which is useful when inspecting label files during testing
        num_classes = scores.shape[1]
        scores = deepcopy(scores[0])
        for i in range(min(num_classes, 10)):
            j = np.argmax(scores)
            logging.info(f"{names[j]}: {scores[j]:.4f}")
            scores[j] = 0

    def save_manifest(self, output_path: str, cfg=None):
        """
        Save a YAML file summarizing the inference configuration.
        """
        from pathlib import Path
        import yaml
        from britekit.models.base_model import BaseModel

        # Add class list
        model: BaseModel = self.models[0]
        names = model.train_class_names
        codes = model.train_class_codes

        info: Dict[str, Any] = {}
        classes = []
        for i, name in enumerate(names):
            classes.append({"name": name, "code": codes[i]})
        info["classes"] = classes

        # Add current inference config
        if cfg is None:
            cfg = self.cfg

        if hasattr(cfg, "audio"):
            info["audio"] = util.cfg_to_pure(cfg.audio)

        if hasattr(cfg, "infer"):
            info["inference"] = util.cfg_to_pure(cfg.infer)

        # Add config per model
        for i, model in enumerate(self.models):
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
            global_frames: (T_global, num_classes) array of scores in [0, 1].
        """
        import numpy as np

        assert (
            frame_scores.ndim == 3
        ), "frame_scores must be (num_specs, num_classes, T_spec)"
        num_specs, num_classes, T_spec = frame_scores.shape

        fps = self.cfg.train.sed_fps
        if fps <= 0:
            raise InferenceError(f"Invalid fps value: {fps}")

        # Global grid length (frames)
        T_global = int(round(fps * recording_duration_sec))

        # Handle edge cases
        if num_specs == 0 or T_global == 0:
            return np.zeros((0, num_classes), dtype=np.float32)

        # Map spectrogram offsets (seconds) to global frame indices
        starts = np.rint(np.array(offsets_sec) * fps).astype(np.int64)

        # Prepare accumulation buffers (float64 for precision)
        global_scores = np.zeros((T_global, num_classes), dtype=np.float64)
        weights = np.zeros((T_global,), dtype=np.float64)

        # Accumulate
        for k in range(num_specs):
            start = starts[k]

            # Skip chunks entirely outside the global range
            if start >= T_global or start + T_spec <= 0:
                continue

            g0, g1 = max(0, start), min(T_global, start + T_spec)
            t0 = g0 - start
            indices = np.arange(g0, g1)
            np.add.at(global_scores, indices, frame_scores[k, :, t0 : t0 + (g1 - g0)].T)
            np.add.at(weights, indices, 1.0)

        # Finalize
        denom = np.clip(weights, 1e-12, None)[:, np.newaxis]
        return (global_scores / denom).astype(np.float32)

    # =============================================================================
    # Private Helper Methods
    # =============================================================================

    def _get_start_times(self, audio_duration, start_seconds, overlap=None):
        """
        Return start offset per spectrogram.

        - audio_duration (float): total audio duration in seconds
        - start_seconds (float): where to start processing the audio (offset in seconds)
        """

        if overlap is None:
            overlap = self.cfg.infer.overlap

        increment = max(0.5, self.cfg.audio.spec_duration - overlap)
        end_offset = max(start_seconds, audio_duration - increment)
        start_times = util.get_range(start_seconds, end_offset, increment)

        # Only keep start times that have at least 1 second of audio
        min_useful_audio = 1.0
        max_start = audio_duration - min_useful_audio
        return [t for t in start_times if t <= max_start]

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

                    # set defaults for missing values (assume name is defined)
                    for i, code in enumerate(self.class_codes):
                        if not code:
                            self.class_codes[i] = self.class_names[i]

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
