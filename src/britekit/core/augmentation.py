#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import ctypes
from functools import partial
import logging
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

            self.augmentations.append((prob, bound))

    @register_augmentation("add_real_noise")
    def add_real_noise(self, spec, prob_fade2=0.3, min_fade2=0.1, max_fade2=0.8):
        """
        Add an actual noise spectrogram but, unlike mixup, do not update the label.
        """
        global _have_real_noise
        if not _have_real_noise.value:
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
    def flip_horizontal(self, spec):
        """
        Flips the spectrogram along the time axis.
        """
        return np.flip(spec, axis=-1)

    @register_augmentation("freq_mask")
    def freq_mask(self, spec, max_width1=8):
        """Mask a random frequency band by setting it to zero."""
        f = spec.shape[-2]
        w = min(np.random.randint(1, max_width1 + 1), f)  # Ensure w doesn't exceed f
        start = np.random.randint(0, max(1, f - w))  # Ensure start is valid
        spec[..., start : start + w, :] = 0
        return spec

    @register_augmentation("shift_horizontal")
    def shift_horizontal(self, spec, max_shift=6):
        """
        Perform a random horizontal shift of the spectrogram
        """
        if max_shift <= 0:
            return spec

        roll_frames = random.randint(-max_shift, max_shift)
        return np.roll(spec, shift=roll_frames, axis=spec.ndim - 1)

    @register_augmentation("speckle")
    def speckle(self, spec, std2=0.1):
        """
        Add a copy multiplied by random pixels (larger stdev leads to more speckling)
        """
        noise = np.random.normal(loc=0.0, scale=std2, size=spec.shape)
        spec += spec * noise
        return np.clip(spec, 0, 1)

    @register_augmentation("time_mask")
    def time_mask(self, spec, max_width2=16):
        """Mask a random time segment by setting it to zero."""
        t = spec.shape[-1]
        w = min(np.random.randint(1, max_width2 + 1), t)  # Ensure w doesn't exceed t
        start = np.random.randint(0, max(1, t - w))  # Ensure start is valid
        spec[..., :, start : start + w] = 0
        return spec

    def __call__(self, spec):
        """
        Apply the augmentation pipeline to a spectrogram.

        Args:
            spec: Input spectrogram to augment

        Returns:
            Augmented spectrogram with values clipped to [0, 1]
        """
        num_augmentations = 0
        for prob, fn in self.augmentations:
            if num_augmentations >= self.cfg.train.max_augmentations:
                break

            if random.random() < prob:
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

        return spec
