#!/usr/bin/env python3

from typing import Optional

import torch


def to_mel(f):
    """
    Convert Hz to mel scale.
    Accepts float or torch.Tensor input.
    """
    if isinstance(f, torch.Tensor):
        return 2595.0 * torch.log10(1.0 + f / 700.0)
    else:
        return 2595.0 * torch.log10(torch.tensor(1.0 + f / 700.0))


def from_mel(m):
    """
    Convert mel scale to Hz.
    Accepts float or torch.Tensor input.
    """
    if isinstance(m, torch.Tensor):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    else:
        return 700.0 * (10.0 ** (torch.tensor(m / 2595.0)) - 1.0)


def band_limited_energy(
    spec: torch.Tensor,
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
    device = spec.device

    if is_mel:
        if f_max is None:
            f_max = sr / 2
        # Mel frequency bin centers
        mel_points = torch.linspace(
            to_mel(f_min), to_mel(f_max), n_freqs, device=device
        )
        freqs = from_mel(mel_points)
    else:
        # Linear frequency bins (e.g., for linear STFT)
        freqs = torch.linspace(0, sr / 2, n_freqs, device=device)

    # Mask for band-limited range
    min_freq, max_freq = freq_range
    band_mask = (freqs >= min_freq) & (freqs <= max_freq)

    # Sum over selected bins and return the log
    energy = spec[band_mask].sum()
    return torch.log10(energy).item()
