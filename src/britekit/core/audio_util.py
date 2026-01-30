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
