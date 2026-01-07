#!/usr/bin/env python3

from typing import Optional

import numpy as np


def to_mel(f):
    """
    Convert Hz to mel scale.
    Accepts float or numpy input.
    """

    # Clip to avoid zero or negative input
    x = np.clip(1.0 + f / 700.0, 1e-10, None)
    return 2595.0 * np.log10(x)


def from_mel(m):
    """
    Convert mel scale to Hz.
    Accepts float or numpy input.
    """
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def band_limited_energy(
    spec: np.ndarray,
    sr: int,
    freq_range: tuple,
    is_mel: bool = True,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> float:
    """
    Compute total energy in a given frequency band from a power spectrogram.

    Args:
        spec (torch.Tensor): Spectrogram tensor of shape (n_freqs, time).
        sr (int): Sampling rate in Hz.
        freq_range (tuple): (min_freq, max_freq) in Hz to extract energy from.
        is_mel (bool): If True, assumes mel scale. If False, assumes linear frequency.
        f_min (float): Minimum frequency used in mel filterbank.
        f_max (float | None): Maximum frequency used in mel filterbank. Defaults to sr / 2.

    Returns:
        torch.Tensor: Scalar tensor representing total energy in the band.
    """
    n_freqs = spec.shape[0]

    if is_mel:
        if f_max is None:
            f_max = sr / 2
        # Mel frequency bin centers
        mel_points = np.linspace(to_mel(f_min), to_mel(f_max), n_freqs)
        freqs = from_mel(mel_points)
    else:
        # Linear frequency bins (e.g., for linear STFT)
        freqs = np.linspace(0, sr / 2, n_freqs)

    # Mask for band-limited range
    min_freq, max_freq = freq_range
    band_mask = (freqs >= min_freq) & (freqs <= max_freq)

    # Sum over selected bins and return the log
    energy = spec[band_mask].sum()

    # Clip to avoid zero or negative input
    return np.log10(np.clip(energy, 1e-10, None))
