#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import logging
import pickle
import random
from typing import Any, List, Optional, Tuple

from lightning.pytorch import LightningDataModule
from torch.utils.data import Sampler

from britekit.core.config_loader import get_config


class PerRecordingSampler(Sampler):
    """
    Samples up to max_per_recording spectrogram indices per recording per epoch.
    Samples belonging to classes with fewer than min_recordings distinct recordings
    are always retained. Other indices are drawn randomly and the final list is
    shuffled.
    """

    def __init__(
        self,
        indices: List[int],
        recording_ids: List[int],
        class_indexes: List[List[int]],
        max_per_recording: int,
        min_recordings: Optional[int] = None,
    ):
        if max_per_recording <= 0:
            raise ValueError("max_per_recording must be positive")
        if min_recordings is not None and min_recordings <= 0:
            raise ValueError("max_per_recording_min_recordings must be positive")
        if len(recording_ids) != len(class_indexes):
            raise ValueError("recording_ids and class_indexes must have equal lengths")

        self.indices = indices
        self.recording_ids = recording_ids
        self.class_indexes = class_indexes
        self.max_per_recording = max_per_recording

        # Group subset indices by recording ID
        groups: dict = {}
        for idx in indices:
            rec_id = recording_ids[idx]
            groups.setdefault(rec_id, []).append(idx)
        self.groups = list(groups.values())

        self.protected_classes = set()
        if min_recordings is not None:
            class_recordings: dict = {}
            for idx in indices:
                rec_id = recording_ids[idx]
                for class_index in class_indexes[idx]:
                    class_recordings.setdefault(class_index, set()).add(rec_id)
            self.protected_classes = {
                class_index
                for class_index, recordings in class_recordings.items()
                if len(recordings) < min_recordings
            }

    def __iter__(self):
        selected = []
        for group in self.groups:
            protected = [
                idx
                for idx in group
                if self.protected_classes.intersection(self.class_indexes[idx])
            ]
            unprotected = [idx for idx in group if idx not in protected]
            k = min(self.max_per_recording, len(unprotected))
            selected.extend(protected)
            selected.extend(random.sample(unprotected, k))
        random.shuffle(selected)
        return iter(selected)

    def __len__(self):
        total = 0
        for group in self.groups:
            protected_count = sum(
                bool(self.protected_classes.intersection(self.class_indexes[idx]))
                for idx in group
            )
            total += protected_count + min(
                self.max_per_recording, len(group) - protected_count
            )
        return total


class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        from britekit.core.dataset import SpectrogramDataset

        self.cfg = get_config()
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Load training data
        try:
            (
                class_names,
                class_codes,
                alt_names,
                alt_codes,
                specs,
                labels,
                segment_ids,
                recording_ids,
            ) = self._load_pickle_data(self.cfg.train.train_pickle)

            # Validate loaded data
            if not class_names or not specs or not labels:
                raise ValueError("Training data is empty or invalid")

            if len(specs) != len(labels):
                raise ValueError(
                    f"Mismatch between specs ({len(specs)}) and labels ({len(labels)}) lengths"
                )
            if recording_ids is not None and len(recording_ids) != len(specs):
                raise ValueError(
                    "Mismatch between specs "
                    f"({len(specs)}) and recording IDs ({len(recording_ids)}) lengths"
                )

            self.train_class_names = class_names
            self.train_class_codes = class_codes
            self.train_class_alt_names = alt_names
            self.train_class_alt_codes = alt_codes
            self.specs = specs
            self.labels = labels
            self.segment_ids = segment_ids
            self.recording_ids = recording_ids
            self.num_train_classes = len(class_names)

            # flatten the labels so [[1, 2], [3]] becomes [1, 2, 3]
            self.flattened_labels = [
                item for sublist in self.labels for item in sublist
            ]

            self.num_train_specs = len(self.specs)
            logging.info(
                f"Fetched {self.num_train_specs} spectrograms for {len(self.train_class_names)} classes"
            )
        except Exception as e:
            logging.error(f"Failed to load training data: {e}")
            raise

        if (
            self.cfg.train.noise_class_name
            and self.cfg.train.noise_class_name in self.train_class_names
        ):
            noise_class_index = self.train_class_names.index(
                self.cfg.train.noise_class_name
            )
        else:
            noise_class_index = -1

        # Load frame-label pickle if configured
        frame_label_dict = None
        if self.cfg.train.frame_label_pickle:
            try:
                with open(self.cfg.train.frame_label_pickle, "rb") as f:
                    frame_label_dict = pickle.load(f)
                logging.info(
                    f"Loaded frame labels for {len(frame_label_dict)} segments "
                    f"from {self.cfg.train.frame_label_pickle}"
                )
            except Exception as e:
                logging.warning(f"Failed to load frame-label pickle: {e}. Ignoring.")

        teacher_targets = None
        teacher_frame_targets = None
        if (
            self.cfg.train.teacher_only_if_no_frame
            and not self.cfg.train.teacher_targets_pickle
        ):
            raise ValueError("teacher_only_if_no_frame requires teacher_targets_pickle")
        if self.cfg.train.teacher_targets_pickle:
            teacher_targets, teacher_frame_targets = self._load_teacher_targets(
                self.cfg.train.teacher_targets_pickle,
                self.train_class_codes,
                self.segment_ids,
            )

        self.full_dataset = SpectrogramDataset(
            self.specs,
            self.labels,
            len(self.train_class_names),
            noise_class_index,
            is_training=True,
            segment_ids=self.segment_ids,
            recording_ids=self.recording_ids,
            frame_label_dict=frame_label_dict,
            teacher_targets=teacher_targets,
            teacher_frame_targets=teacher_frame_targets,
        )

        # Load test data
        if self.cfg.train.test_pickle:
            try:
                class_names, class_codes, alt_names, alt_codes, specs, labels, _, _ = (
                    self._load_pickle_data(self.cfg.train.test_pickle)
                )

                # Validate test data
                if not class_names or not specs or not labels:
                    logging.error(
                        "Test data is empty or invalid, setting test_data to None"
                    )
                    self.test_data = None
                elif len(specs) != len(labels):
                    logging.error(
                        f"Mismatch between test specs ({len(specs)}) and labels ({len(labels)}) lengths, setting test_data to None"
                    )
                    self.test_data = None
                else:
                    self.test_specs = specs
                    self.test_labels = labels

                    self.test_data = SpectrogramDataset(
                        self.test_specs,
                        self.test_labels,
                        len(class_names),
                        is_training=False,
                    )
            except Exception as e:
                logging.error(
                    f"Failed to load test data: {e}, setting test_data to None"
                )
                self.test_data = None
        else:
            self.test_data = None

        if self.cfg.train.num_folds > 1:
            # Stratified k-fold split
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(
                n_splits=self.cfg.train.num_folds, shuffle=True, random_state=42
            )
            # Use first label per segment for stratification (multi-label segments
            # have lists; StratifiedKFold requires a flat 1D array).
            stratify_labels = [lbl[0] if lbl else 0 for lbl in self.labels]
            self.indices = list(skf.split(self.specs, stratify_labels))

    def _load_pickle_data(self, path: str) -> Tuple[
        List[str],
        List[str],
        List[str],
        List[str],
        List[Any],
        List[List[int]],
        Optional[List[int]],
        Optional[List[int]],
    ]:
        """
        Load data from a pickle file with error handling.

        Args:
        - path (str): Path to the pickle file

        Returns:
            Tuple containing (class_names, class_codes, alt_names, alt_codes, specs, labels, segment_ids)

        Raises:
            FileNotFoundError: If the pickle file doesn't exist
            ValueError: If the pickle file is corrupted or missing required keys
        """
        if not path:
            raise ValueError("Pickle file path cannot be empty")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found: {path}")
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Failed to load pickle file {path}: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading pickle file {path}: {e}")

        # Validate required keys exist
        required_keys = [
            "class_names",
            "class_codes",
            "alt_names",
            "alt_codes",
            "spec_values",
            "spec_class_indexes",
        ]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(
                f"Pickle file {path} missing required keys: {missing_keys}"
            )

        # segment_ids and recording_ids were added later; old pickles may not have them
        segment_ids = data.get("spec_segment_ids", None)
        recording_ids = data.get("spec_recording_ids", None)

        return (
            data["class_names"],
            data["class_codes"],
            data["alt_names"],
            data["alt_codes"],
            data["spec_values"],
            data["spec_class_indexes"],
            segment_ids,
            recording_ids,
        )

    def _load_teacher_targets(self, path, class_codes, segment_ids):
        """Load soft targets and reorder them to match the training pickle."""
        import numpy as np

        if segment_ids is None:
            raise ValueError(
                "Training pickle must contain segment IDs when distillation is enabled"
            )
        with open(path, "rb") as file:
            data = pickle.load(file)

        required = ("format_version", "class_codes", "segment_ids", "probabilities")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Teacher-target pickle missing required keys: {missing}")
        if data["format_version"] != 2:
            raise ValueError(
                f"Unsupported teacher-target format version: {data['format_version']}"
            )
        if list(data["class_codes"]) != list(class_codes):
            raise ValueError(
                "Teacher-target class codes and ordering do not match training data"
            )

        target_ids = list(data["segment_ids"])
        probabilities = np.asarray(data["probabilities"], dtype=np.float32)
        expected_shape = (len(target_ids), len(class_codes))
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"Teacher probabilities have shape {probabilities.shape}, expected {expected_shape}"
            )
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("Teacher probabilities must be finite and between 0 and 1")
        if "frame_probabilities" not in data:
            raise ValueError(
                "Teacher-target pickle does not contain frame probabilities"
            )
        frame_probabilities = np.asarray(data["frame_probabilities"])
        if not np.issubdtype(frame_probabilities.dtype, np.floating):
            raise ValueError("Teacher frame probabilities must be floating point")
        expected_frame_prefix = (len(target_ids), len(class_codes))
        if (
            frame_probabilities.ndim != 3
            or frame_probabilities.shape[:2] != expected_frame_prefix
        ):
            raise ValueError(
                "Teacher frame probabilities must have shape "
                f"({len(target_ids)}, {len(class_codes)}, frames), got "
                f"{frame_probabilities.shape}"
            )
        if not np.isfinite(frame_probabilities).all() or np.any(
            (frame_probabilities < 0) | (frame_probabilities > 1)
        ):
            raise ValueError(
                "Teacher frame probabilities must be finite and between 0 and 1"
            )
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("Teacher-target pickle contains duplicate segment IDs")

        target_indexes = {segment_id: i for i, segment_id in enumerate(target_ids)}
        missing_ids = [
            segment_id for segment_id in segment_ids if segment_id not in target_indexes
        ]
        if missing_ids:
            raise ValueError(
                f"Teacher targets are missing {len(missing_ids)} training segment IDs"
            )

        if target_ids == list(segment_ids):
            ordered = probabilities
            ordered_frames = frame_probabilities
        else:
            order = [target_indexes[segment_id] for segment_id in segment_ids]
            ordered = probabilities[order]
            ordered_frames = frame_probabilities[order]
        logging.info(
            "Loaded teacher targets with segment shape %s and frame shape %s from %s",
            ordered.shape,
            ordered_frames.shape,
            path,
        )
        return ordered, ordered_frames

    def class_weights(self):
        import numpy as np

        if self.cfg.train.use_class_weights:
            import sklearn.utils.class_weight

            class_weights = sklearn.utils.class_weight.compute_class_weight(
                "balanced",
                classes=np.arange(self.num_train_classes),
                y=self.flattened_labels,
            )
            class_weights = class_weights**self.cfg.train.weight_exponent
        else:
            class_weights = np.ones(self.num_train_classes)

        return class_weights

    @staticmethod
    def _pin_memory():
        """Pin loader batches only when CUDA can use the faster transfer path."""
        import torch

        return torch.cuda.is_available()

    def _make_val_dataset(self, val_indices):
        """Create a non-augmenting dataset containing only the validation samples."""
        from britekit.core.dataset import SpectrogramDataset

        specs = [self.specs[i] for i in val_indices]
        labels = [self.labels[i] for i in val_indices]
        segment_ids = (
            [self.segment_ids[i] for i in val_indices] if self.segment_ids else None
        )
        recording_ids = (
            [self.recording_ids[i] for i in val_indices] if self.recording_ids else None
        )
        noise_class_index = self.full_dataset.noise_class_index
        return SpectrogramDataset(
            specs,
            labels,
            self.num_train_classes,
            noise_class_index,
            is_training=False,
            segment_ids=segment_ids,
            recording_ids=recording_ids,
        )

    def prepare_fold(self, fold_index: int):
        """
        Prepare train/validation split for a specific fold.

        Args:
        - fold_index (int): Index of the fold to prepare

        Raises:
            ValueError: If fold_index is invalid or val_portion is invalid
        """
        from torch.utils.data import Subset

        if not hasattr(self, "full_dataset") or self.full_dataset is None:
            raise ValueError("Full dataset not initialized")

        if self.cfg.train.num_folds <= 1:
            # Simple train/val split
            if not (0 <= self.cfg.train.val_portion < 1):
                raise ValueError(
                    f"val_portion must be between 0 and 1, got {self.cfg.train.val_portion}"
                )

            val_size = int(len(self.full_dataset) * self.cfg.train.val_portion)
            train_size = len(self.full_dataset) - val_size

            if train_size <= 0:
                raise ValueError(
                    f"Invalid split sizes: train_size={train_size}, val_size={val_size}"
                )

            indices = list(range(len(self.full_dataset)))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            self.train_data = Subset(self.full_dataset, train_indices)
            if val_indices:
                self.val_data = Subset(
                    self._make_val_dataset(val_indices), list(range(len(val_indices)))
                )
            else:
                self.val_data = None
        else:
            # Stratified k-fold split
            if not hasattr(self, "indices") or not self.indices:
                raise ValueError("K-fold indices not initialized")

            if fold_index < 0 or fold_index >= len(self.indices):
                raise ValueError(
                    f"fold_index {fold_index} out of range [0, {len(self.indices)})"
                )

            train_idx, val_idx = self.indices[fold_index]
            self.train_data = Subset(self.full_dataset, train_idx)
            self.val_data = Subset(
                self._make_val_dataset(val_idx), list(range(len(val_idx)))
            )

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        if self.train_data is None:
            raise ValueError("Training data not prepared. Call prepare_fold() first.")

        max_per_recording = self.cfg.train.max_per_recording
        if max_per_recording is not None:
            if self.recording_ids is None:
                raise ValueError(
                    "max_per_recording requires a training pickle containing "
                    "recording IDs"
                )
            # Subset indices are original-dataset indices; sampler must output
            # subset-relative indices (0..len(subset)-1) for the DataLoader.
            train_indices = list(self.train_data.indices)
            subset_recording_ids = [self.recording_ids[i] for i in train_indices]
            subset_class_indexes = [self.labels[i] for i in train_indices]
            sampler = PerRecordingSampler(
                list(range(len(train_indices))),
                subset_recording_ids,
                subset_class_indexes,
                max_per_recording,
                self.cfg.train.max_per_recording_min_recordings,
            )
            return DataLoader(
                self.train_data,
                batch_size=self.cfg.train.batch_size,
                sampler=sampler,
                num_workers=self.cfg.train.num_workers,
                pin_memory=self._pin_memory(),
            )

        return DataLoader(
            self.train_data,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.train.shuffle,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self._pin_memory(),
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        if self.val_data is None:
            return None

        val_max = self.cfg.train.val_max_per_recording
        val_dataset = self.val_data.dataset  # the non-augmenting SpectrogramDataset
        if val_max is not None and val_dataset.recording_ids is not None:
            # Take first val_max segments per recording (deterministic).
            groups: dict = {}
            for idx in range(len(val_dataset)):
                rec_id = val_dataset.recording_ids[idx]
                if rec_id not in groups:
                    groups[rec_id] = []
                if len(groups[rec_id]) < val_max:
                    groups[rec_id].append(idx)
            selected = [idx for group in groups.values() for idx in group]
            return DataLoader(
                val_dataset,
                batch_size=self.cfg.train.batch_size,
                sampler=selected,
                num_workers=self.cfg.train.num_workers,
                pin_memory=self._pin_memory(),
            )

        return DataLoader(
            self.val_data,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self._pin_memory(),
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        if self.test_data is None:
            logging.error("Test data not available, returning None")
            return None
        return DataLoader(
            self.test_data,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self._pin_memory(),
        )
