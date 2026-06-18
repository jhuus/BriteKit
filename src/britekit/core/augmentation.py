#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import ctypes
from functools import partial
import logging
import math
from multiprocessing import Value
import random

import numpy as np
from scipy.ndimage import gaussian_filter

from britekit.core.base_config import BaseConfig

AUGMENTATION_REGISTRY = {}
_have_real_noise = Value(ctypes.c_bool, True)


def register_augmentation(name):
    """Decorator to register an augmentation function in the global registry."""

    def decorator(fn):
        AUGMENTATION_REGISTRY[name] = fn
        return fn

    return decorator


class AugmentationPipeline:
    """Pipeline for applying audio spectrogram augmentations during training."""

    def __init__(self, cfg: BaseConfig, dataset):
        """
        Initialize the augmentation pipeline with configuration and dataset.

        Args:
            cfg: Configuration object containing augmentation settings
            dataset: Dataset object for accessing noise samples
        """
        self.cfg = cfg
        self.dataset = dataset
        self.augmentations = []

        if not cfg.train.augmentations:
            return  # No augmentations to configure

        for aug_cfg in cfg.train.augmentations:
            if "name" not in aug_cfg:
                raise ValueError("Augmentation config missing required 'name' key")

            name = aug_cfg["name"]
            prob = aug_cfg.get("prob", 1.0)
            params = aug_cfg.get("params", {})

            if name not in AUGMENTATION_REGISTRY:
                raise ValueError(f"Unknown augmentation: {name}")

            # get unbound function and bind it to self
            fn_unbound = AUGMENTATION_REGISTRY[name]
            bound = fn_unbound.__get__(self, self.__class__)

            if params:
                bound = partial(bound, **params)

            self.augmentations.append((name, prob, bound))

    def _scale_time_index(self, pixel_index, spec_frames, label_frames, rounding):
        scaled = pixel_index * label_frames / spec_frames
        if rounding == "floor":
            return math.floor(scaled)
        if rounding == "ceil":
            return math.ceil(scaled)
        return round(scaled)

    def _roll_frame_labels(self, frame_labels, shift):
        if shift == 0:
            return frame_labels
        if hasattr(frame_labels, "roll"):
            return frame_labels.roll(shifts=shift, dims=0)
        return np.roll(frame_labels, shift=shift, axis=0)

    def _shift_frame_labels(self, frame_labels, shift):
        if shift == 0:
            return frame_labels

        if hasattr(frame_labels, "new_zeros"):
            result = frame_labels.new_zeros(frame_labels.shape)
        else:
            result = np.zeros_like(frame_labels)

        if abs(shift) >= frame_labels.shape[0]:
            return result

        if shift > 0:
            result[shift:] = frame_labels[:-shift]
        else:
            shift = -shift
            result[:-shift] = frame_labels[shift:]
        return result

    def _mask_frame_labels(self, frame_labels, start, end):
        if end <= start:
            return frame_labels

        if hasattr(frame_labels, "clone"):
            result = frame_labels.clone()
        else:
            result = frame_labels.copy()
        result[start:end] = 0
        return result

    @register_augmentation("add_real_noise")
    def add_real_noise(self, spec, prob_fade2=0.3, min_fade2=0.1, max_fade2=0.8):
        """
        Add an actual noise spectrogram but, unlike mixup, do not update the label.
        """
        global _have_real_noise
        if not _have_real_noise.value or self.dataset is None:
            return spec

        noise_spec = self.dataset.get_random_noise()
        if noise_spec is None:
            # with multiple workers, only do this once
            with _have_real_noise.get_lock():
                if _have_real_noise.value:
                    _have_real_noise.value = False
                    logging.error("")
                    logging.error("*** WARNING:")
                    logging.error(
                        "No noise class is defined, but add_real_noise is enabled."
                    )
                    logging.error("In most cases it is best to provide noise data.")
                    logging.error(
                        "The add_real_noise augmentation will be disabled in this run."
                    )
                    logging.error("")
            return spec

        # Validate shapes match
        if noise_spec.shape != spec.shape:
            raise ValueError(
                f"Shape mismatch: spec {spec.shape} vs noise {noise_spec.shape}"
            )

        # fade the spec sometimes
        if random.random() < prob_fade2:
            spec *= random.uniform(min_fade2, max_fade2)

        spec += noise_spec
        return spec

    @register_augmentation("add_white_noise")
    def add_white_noise(self, spec, min_std=0.01, max_std=0.1, max_val=2.5):
        """Add Gaussian white noise to the spectrogram."""

        std = random.uniform(min_std, max_std)
        noise = np.abs(np.random.normal(0, std, size=spec.shape))
        noise = np.clip(noise / max_val, 0, 1)
        return np.clip(spec + noise, 0.0, 1.0)

    @register_augmentation("blur")
    def blur(self, spec, min_sigma=0.0, max_sigma=1.0):
        """Apply Gaussian blur equally along both axes."""
        sigma = random.uniform(min_sigma, max_sigma)
        return gaussian_filter(spec, sigma=sigma)

    @register_augmentation("flip_horizontal")
    def flip_horizontal(self, spec, frame_labels=None):
        """
        Flips the spectrogram along the time axis.
        """
        spec = np.flip(spec, axis=-1)
        if frame_labels is None:
            return spec
        if hasattr(frame_labels, "flip"):
            return spec, frame_labels.flip(0)
        return spec, np.flip(frame_labels, axis=0)

    @register_augmentation("freq_mask")
    def freq_mask(self, spec, max_width1=8, num_masks1=1):
        """Mask random frequency bands by setting them to zero."""
        f = spec.shape[-2]
        for _ in range(num_masks1):
            w = min(np.random.randint(1, max_width1 + 1), f)
            start = np.random.randint(0, f - w + 1)
            spec[..., start : start + w, :] = 0
        return spec

    @register_augmentation("shift_horizontal")
    def shift_horizontal(self, spec, max_shift=6, pad_value=None, frame_labels=None):
        """
        Random horizontal shift. If pad_value is None, wrap.
        Otherwise fill newly exposed frames with pad_value.
        """
        if max_shift <= 0:
            return spec if frame_labels is None else (spec, frame_labels)

        spec_frames = spec.shape[-1]

        if pad_value is None:
            # do a roll
            roll_frames = random.randint(-max_shift, max_shift)
            spec = np.roll(spec, shift=roll_frames, axis=spec.ndim - 1)
            if frame_labels is None:
                return spec
            label_shift = self._scale_time_index(
                roll_frames, spec_frames, frame_labels.shape[0], "round"
            )
            return spec, self._roll_frame_labels(frame_labels, label_shift)

        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return spec if frame_labels is None else (spec, frame_labels)

        axis = spec.ndim - 1
        result = np.full_like(spec, pad_value)

        if shift > 0:
            # shift right
            src = [slice(None)] * spec.ndim
            dst = [slice(None)] * spec.ndim
            src[axis] = slice(0, -shift)
            dst[axis] = slice(shift, None)
            result[tuple(dst)] = spec[tuple(src)]
        else:
            # shift left
            shift = -shift
            src = [slice(None)] * spec.ndim
            dst = [slice(None)] * spec.ndim
            src[axis] = slice(shift, None)
            dst[axis] = slice(0, -shift)
            result[tuple(dst)] = spec[tuple(src)]
            shift = -shift

        if frame_labels is None:
            return result

        label_shift = self._scale_time_index(
            shift, spec_frames, frame_labels.shape[0], "round"
        )
        return result, self._shift_frame_labels(frame_labels, label_shift)

    @register_augmentation("speckle")
    def speckle(self, spec, std2=0.1):
        """
        Add a copy multiplied by random pixels (larger stdev leads to more speckling)
        """
        noise = np.random.normal(loc=0.0, scale=std2, size=spec.shape)
        spec += spec * noise
        return np.clip(spec, 0, 1)

    @register_augmentation("time_mask")
    def time_mask(self, spec, max_width2=16, num_masks2=1, frame_labels=None):
        """Mask random time segments by setting them to zero."""
        t = spec.shape[-1]
        for _ in range(num_masks2):
            w = min(np.random.randint(1, max_width2 + 1), t)
            start = np.random.randint(0, t - w + 1)
            spec[..., :, start : start + w] = 0
            if frame_labels is not None:
                label_start = self._scale_time_index(
                    start, t, frame_labels.shape[0], "floor"
                )
                label_end = self._scale_time_index(
                    start + w, t, frame_labels.shape[0], "ceil"
                )
                frame_labels = self._mask_frame_labels(
                    frame_labels, label_start, label_end
                )
        return spec if frame_labels is None else (spec, frame_labels)

    def __call__(self, spec, frame_labels=None):
        """
        Apply the augmentation pipeline to a spectrogram.

        Args:
            spec: Input spectrogram to augment

        Returns:
            Augmented spectrogram with values clipped to [0, 1]
        """
        num_augmentations = 0
        for name, prob, fn in self.augmentations:
            if num_augmentations >= self.cfg.train.max_augmentations:
                break

            if random.random() < prob:
                if frame_labels is not None and name in (
                    "flip_horizontal",
                    "shift_horizontal",
                    "time_mask",
                ):
                    spec, frame_labels = fn(spec, frame_labels=frame_labels)
                else:
                    spec = fn(spec)
                num_augmentations += 1

        # set max value = 1
        max_val = spec.max()
        if max_val > 0 and not np.isnan(max_val):
            spec = spec / max_val

        spec = spec.clip(0, 1)  # in case there are negative values

        # reducing the max level after normalization improves detection of faint sounds
        if random.random() < self.cfg.train.prob_fade1:
            spec *= random.uniform(self.cfg.train.min_fade1, self.cfg.train.max_fade1)

        return spec if frame_labels is None else (spec, frame_labels)
