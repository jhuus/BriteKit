#!/usr/bin/env python3

import random

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional

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
        if self.is_training and self.augment:
            if (
                self.cfg.train.multi_label
                and self.class_indexes[idx][0] != self.noise_class_index
                and random.random() < self.cfg.train.prob_simple_merge
            ):
                spec, label_tensor = self._merge_specs(
                    spec, label_tensor, self.class_indexes[idx]
                )
                mixup = True

            spec = self.augment(spec)

        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        mixup = torch.tensor(mixup)

        return {
            "input": spec_tensor,
            "segment_labels": label_tensor,
            "mixup": mixup,
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

    def _get_spec(self, idx):
        # Validate index bounds
        if idx < 0 or idx >= len(self.compressed_specs):
            raise IndexError(
                f"Index {idx} out of bounds [0, {len(self.compressed_specs)})"
            )

        spec = expand_spectrogram(self.compressed_specs[idx])
        return spec.reshape(1, self.cfg.audio.spec_height, self.cfg.audio.spec_width)

    def _merge_specs(self, spec, label_tensor, class_indexes):
        """
        Given a spectrogram and label, pick another random non-noise spec
        and return the sum of the two, and the sum of their labels.
        """

        # pick a non-noise spectrogram from a different class
        while True:
            other_index = random.randint(0, len(self.class_indexes) - 1)
            other_class_index = self.class_indexes[other_index][0]
            if (
                other_class_index not in class_indexes
                and other_class_index != self.noise_class_index
            ):
                break

        other_spec = self._get_spec(other_index)
        other_label_tensor = torch.nn.functional.one_hot(
            torch.tensor(other_class_index, dtype=torch.long),
            num_classes=self.num_classes,
        ).float()

        return spec + other_spec, label_tensor + other_label_tensor
