#!/usr/bin/env python3

import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from britekit.core.augmentation import AugmentationPipeline
from britekit.core.config_loader import get_config
from britekit.core.util import expand_spectrogram


class SpectrogramDataset(Dataset):
    """
    Training dataset created from pickled data.

    Attributes:
        compressed_specs: List of compressed spectrograms.
        class_indexes: Element i has list of class indexes for spectrogram i,
                since a spectrogram may have more than one class.
        num_classes: Number of classes.
        noise_class_index: Index of the noise class, or -1 if not defined.
        is_training: True if training, else false.
    """

    def __init__(
        self,
        compressed_specs: List[Any],
        class_indexes: List[List[int]],
        num_classes: int,
        noise_class_index: int = -1,
        is_training: bool = True,
        segment_ids: Optional[List[int]] = None,
        recording_ids: Optional[List[int]] = None,
        frame_label_dict: Optional[Dict[int, np.ndarray]] = None,
    ):
        # Input validation
        if not compressed_specs or not class_indexes:
            raise ValueError("compressed_specs and class_indexes cannot be empty")

        if len(compressed_specs) != len(class_indexes):
            raise ValueError(
                f"Mismatch between compressed_specs ({len(compressed_specs)}) and class_indexes ({len(class_indexes)}) lengths"
            )

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        # Validate class indexes are within bounds
        for i, indexes in enumerate(class_indexes):
            if not indexes:  # Empty list is valid (no classes)
                continue
            for class_idx in indexes:
                if class_idx < 0 or class_idx >= num_classes:
                    raise ValueError(
                        f"Class index {class_idx} at position {i} is out of bounds [0, {num_classes})"
                    )

        self.compressed_specs = compressed_specs
        self.class_indexes = class_indexes
        self.num_classes = num_classes
        self.noise_class_index = noise_class_index
        self.is_training = is_training
        self.segment_ids = segment_ids
        self.recording_ids = recording_ids
        self.frame_label_dict = frame_label_dict

        self.cfg = get_config()

        # get indexes of all specs that contain only noise
        self.noise_indexes = []
        for i, item in enumerate(class_indexes):
            if len(item) == 1 and item[0] == noise_class_index:
                self.noise_indexes.append(i)

        self.augment: Optional[Callable] = None
        if is_training and self.cfg.train.augment:
            self.augment = AugmentationPipeline(self.cfg, self)

    def __len__(self):
        return len(self.compressed_specs)

    def __getitem__(self, idx):
        # Validate index bounds
        if idx < 0 or idx >= len(self.compressed_specs):
            raise IndexError(
                f"Index {idx} out of bounds [0, {len(self.compressed_specs)})"
            )

        spec = self._get_spec(idx)

        label_indexes = self.class_indexes[idx]
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        label_tensor[label_indexes] = 1.0

        mixup = False
        cutmix_info = (
            None  # (time_range, other_label, original_label) if cutmix was applied
        )
        if self.is_training and self.augment:
            if (
                self.cfg.train.multi_label
                and self.class_indexes[idx][0] != self.noise_class_index
            ):
                r = random.random()
                if r < self.cfg.train.prob_simple_merge:
                    spec, label_tensor = self._merge_specs(
                        spec, label_tensor, self.class_indexes[idx]
                    )
                    mixup = True
                elif r < self.cfg.train.prob_simple_merge + self.cfg.train.prob_mixup:
                    spec, label_tensor = self._mixup(
                        spec, label_tensor, self.class_indexes[idx]
                    )
                    mixup = True
                elif (
                    r
                    < self.cfg.train.prob_simple_merge
                    + self.cfg.train.prob_mixup
                    + self.cfg.train.prob_cutmix
                ):
                    original_label = label_tensor.clone()
                    spec, label_tensor, time_range, other_label = self._cutmix(
                        spec, label_tensor, self.class_indexes[idx]
                    )
                    cutmix_info = (time_range, other_label, original_label)
                    mixup = True

            spec = self.augment(spec)

        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        frame_labels = self._get_frame_labels(idx, label_tensor, mixup, cutmix_info)
        mixup = torch.tensor(mixup)

        return {
            "input": spec_tensor,
            "segment_labels": label_tensor,
            "mixup": mixup,
            "frame_labels": frame_labels,
        }

    def get_random_noise(self):
        """
        Return a random noise spec from the training data
        """
        if not self.noise_indexes:
            return None

        idx = random.randint(0, len(self.noise_indexes) - 1)
        return self._get_spec(self.noise_indexes[idx])

    # =============================================================================
    # Private Helper Methods
    # =============================================================================

    def _get_frame_labels(self, idx, label_tensor, mixup, cutmix_info=None):
        """
        Return frame-level labels as a (num_frames, num_classes) float32 tensor,
        where num_frames = round(spec_duration * sed_fps).

        For CutMix, assigns labelA to frames outside the pasted region and labelB
        to frames inside it, giving accurate per-frame supervision.

        If stored frame labels are available for this segment and no mixup occurred,
        use them for the present class. Otherwise fall back to broadcasting
        the segment label to all frames (all-ones behaviour).
        """
        num_frames = round(self.cfg.audio.spec_duration * self.cfg.train.sed_fps)

        if cutmix_info is not None:
            time_range, other_label, original_label = cutmix_info
            y1, y2 = time_range
            frame_start = max(0, round(y1 * num_frames / self.cfg.audio.spec_width))
            frame_end = min(
                num_frames, round(y2 * num_frames / self.cfg.audio.spec_width)
            )
            frame_labels = original_label.unsqueeze(0).expand(num_frames, -1).clone()
            if frame_end > frame_start:
                frame_labels[frame_start:frame_end] = other_label
            return frame_labels

        stored = None
        if (
            not mixup
            and self.frame_label_dict is not None
            and self.segment_ids is not None
        ):
            seg_id = self.segment_ids[idx]
            stored = self.frame_label_dict.get(seg_id)  # (num_frames,) or None

        present = (label_tensor > 0).nonzero(as_tuple=True)[0]
        if stored is not None and len(present) == 1:
            # Single-class segment: apply stored frame pattern to the present class.
            frame_labels = torch.zeros(
                num_frames, self.num_classes, dtype=torch.float32
            )
            stored_tensor = torch.from_numpy(stored)
            frame_labels[:, present[0]] = stored_tensor
        else:
            # Multi-class or no stored labels: broadcast segment labels to all frames.
            frame_labels = label_tensor.unsqueeze(0).expand(num_frames, -1).clone()

        return frame_labels

    def _get_spec(self, idx):
        # Validate index bounds
        if idx < 0 or idx >= len(self.compressed_specs):
            raise IndexError(
                f"Index {idx} out of bounds [0, {len(self.compressed_specs)})"
            )

        spec = expand_spectrogram(self.compressed_specs[idx])
        return spec.reshape(1, self.cfg.audio.spec_height, self.cfg.audio.spec_width)

    def _mixup(self, spec, label_tensor, class_indexes):
        """
        Traditional mixup: blend two spectrograms and their labels with a lambda
        drawn from Beta(alpha, alpha). Unlike simple merge, this produces a
        weighted combination rather than a sum, so soft labels are used.
        """
        class_index_set = set(class_indexes)
        while True:
            other_index = random.randint(0, len(self.class_indexes) - 1)
            other_indexes = self.class_indexes[other_index]
            if (
                not any(i in class_index_set for i in other_indexes)
                and self.noise_class_index not in other_indexes
            ):
                break

        lam = np.random.beta(self.cfg.train.mixup_alpha, self.cfg.train.mixup_alpha)
        other_spec = self._get_spec(other_index)
        other_label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        other_label_tensor[other_indexes] = 1.0

        return (
            lam * spec + (1 - lam) * other_spec,
            lam * label_tensor + (1 - lam) * other_label_tensor,
        )

    def _cutmix(self, spec, label_tensor, class_indexes):
        """
        CutMix: paste a rectangular region from another spectrogram into this one.
        Lambda is drawn from Beta(alpha, alpha) and determines the box size via
        cut_ratio = sqrt(1 - lambda). Labels are mixed proportionally to the actual
        pasted area. Returns the mixed spec, mixed segment label, the pasted time
        range in spectrogram pixels (for frame label computation), and the other
        sample's label.
        """
        class_index_set = set(class_indexes)
        while True:
            other_index = random.randint(0, len(self.class_indexes) - 1)
            other_indexes = self.class_indexes[other_index]
            if (
                not any(i in class_index_set for i in other_indexes)
                and self.noise_class_index not in other_indexes
            ):
                break

        lam = np.random.beta(self.cfg.train.mixup_alpha, self.cfg.train.mixup_alpha)
        other_spec = self._get_spec(other_index)
        other_label = torch.zeros(self.num_classes, dtype=torch.float)
        other_label[other_indexes] = 1.0

        _, freq, time = spec.shape
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(freq * cut_ratio)
        cut_w = int(time * cut_ratio)

        cx = np.random.randint(freq)
        cy = np.random.randint(time)
        x1 = max(0, cx - cut_h // 2)
        x2 = min(freq, cx + cut_h // 2)
        y1 = max(0, cy - cut_w // 2)
        y2 = min(time, cy + cut_w // 2)

        spec[:, x1:x2, y1:y2] = other_spec[:, x1:x2, y1:y2]

        actual_lam = 1 - (x2 - x1) * (y2 - y1) / (freq * time)
        mixed_label = actual_lam * label_tensor + (1 - actual_lam) * other_label

        return spec, mixed_label, (y1, y2), other_label

    def _merge_specs(self, spec, label_tensor, class_indexes):
        """
        Given a spectrogram and label, pick another random non-noise spec
        and return the sum of the two, and the sum of their labels.
        """

        # pick a non-noise spectrogram from a different class
        class_index_set = set(class_indexes)
        while True:
            other_index = random.randint(0, len(self.class_indexes) - 1)
            other_indexes = self.class_indexes[other_index]
            if (
                not any(i in class_index_set for i in other_indexes)
                and self.noise_class_index not in other_indexes
            ):
                break

        other_spec = self._get_spec(other_index)
        other_label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        other_label_tensor[other_indexes] = 1.0

        return spec + other_spec, label_tensor + other_label_tensor
