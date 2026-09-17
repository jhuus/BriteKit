#!/usr/bin/env python3

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


def validate_db_power(db_power):
    """Validate the post-normalization exponent for converted linear features."""
    if not np.isfinite(db_power) or db_power <= 0:
        raise ValueError("db_power must be finite and positive")


def convert_to_db(specs, power=1.0, top_db=80.0, normalize=True, db_power=1.0):
    """Convert nonnegative linear features to minmax-normalized dB.

    Conversion operates per final two axes. Accepts magnitude (power=1) or
    power (power=2) features. A relative floor makes uniform gain cancel.
    Constant and silent inputs map to zero. Never modifies the input.
    Set normalize=False for relative dB (peak 0, floor -top_db).
    db_power applies after normalization, never to the raw dB output.
    """
    validate_db_power(db_power)
    if power not in (1.0, 2.0):
        raise ValueError("convert_to_db requires audio.power to be 1 or 2")
    if not np.isfinite(top_db) or top_db <= 0:
        raise ValueError("convert_to_db requires a finite positive top_db")
    values = np.asarray(specs, dtype=np.float32)
    if values.ndim < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("convert_to_db requires finite nonnegative spectrograms")
    peak = values.max(axis=(-2, -1), keepdims=True)
    relative = values / np.maximum(peak, np.finfo(np.float32).tiny)
    multiplier = 20.0 / power
    db = multiplier * np.log10(np.maximum(relative, 10.0 ** (-top_db / multiplier)))
    db = np.where(peak > 0, db, -top_db)
    if not normalize:
        return db
    return normalize_db(db, db_power=db_power)


def normalize_db(db, db_power=1.0):
    """Minmax-normalize dB per final two axes without modifying the input."""
    validate_db_power(db_power)
    values = np.asarray(db, dtype=np.float32)
    shifted = values - values.min(axis=(-2, -1), keepdims=True)
    span = shifted.max(axis=(-2, -1), keepdims=True)
    normalized = shifted / np.maximum(span, np.finfo(np.float32).tiny)
    return normalized if db_power == 1 else normalized**db_power
