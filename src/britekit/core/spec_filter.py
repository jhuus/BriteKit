"""
SpecFilter applies frequency-domain masks (low-pass,
band-pass, high-pass) to spectrogram tensors.

Note: this class is experimental and not currently used.
"""

from __future__ import annotations

import torch

from britekit.core.base_config import BaseConfig


class SpecFilter:
    """
    Spectrogram frequency filter operating directly on spectrograms.
    """

    def __init__(self, cfg: BaseConfig, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.min_val = torch.tensor(cfg.infer.min_filter_value, device=self.device, dtype=torch.float32)
        self.min_normalizer = torch.tensor(
            self.cfg.infer.min_filter_normalizer, device=self.device, dtype=torch.float32
        )
        self.spec_height = cfg.audio.spec_height

        # Precompute frequency axis
        self._f = torch.linspace(
            0.0, 1.0, self.spec_height, device=self.device, dtype=torch.float32
        ).unsqueeze(
            1
        )  # (H, 1)

        # Precompute masks
        self._lp_mask = self._create_low_pass_mask()
        self._bp_mask = self._create_band_pass_mask()
        self._hp_mask = self._create_high_pass_mask()

    # ------------------------------------------------------------------
    # Mask creation
    # ------------------------------------------------------------------

    def _create_low_pass_mask(self) -> torch.Tensor:
        mask = torch.sigmoid(
            (self.cfg.infer.low_pass_end - self._f) * self.cfg.infer.filter_steepness
        )

        # ensure range is [0, 1], then set min value
        mask -= mask.min()
        mask /= mask.max().clamp_min(1e-6)
        return (mask * (1 - self.min_val) + self.min_val).unsqueeze(0)

    def _create_high_pass_mask(self) -> torch.Tensor:
        mask = torch.sigmoid(
            (self._f - self.cfg.infer.high_pass_start) * self.cfg.infer.filter_steepness
        )

        # ensure range is [0, 1], then set min value
        mask -= mask.min()
        mask /= mask.max().clamp_min(1e-6)
        return (mask * (1 - self.min_val) + self.min_val).unsqueeze(0)

    def _create_band_pass_mask(self) -> torch.Tensor:
        low_q = self.cfg.infer.band_pass_start
        high_q = self.cfg.infer.band_pass_end

        if low_q >= high_q:
            raise ValueError("band-pass start must be < end")

        low_edge = torch.sigmoid((self._f - low_q) * self.cfg.infer.filter_steepness)
        high_edge = torch.sigmoid((high_q - self._f) * self.cfg.infer.filter_steepness)

        mask = low_edge * high_edge

        # ensure range is [0, 1], then set min value
        mask -= mask.min()
        mask /= mask.max().clamp_min(1e-6)
        return (mask * (1 - self.min_val) + self.min_val).unsqueeze(0)

    def _apply_filter(self, spec_array: torch.Tensor, mask: torch.Tensor, normalize: bool = False):
        """
        spec_array: (N, H, W)
        mask: (1, H, 1)
        """
        out = spec_array * mask

        if not normalize:
            return out

        max_vals = out.amax(dim=(1, 2), keepdim=True).clamp_min(self.min_normalizer)
        return out / max_vals


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def low_pass_filter(self, spec_array: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self._apply_filter(spec_array, self._lp_mask, normalize)

    def band_pass_filter(self, spec_array: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self._apply_filter(spec_array, self._bp_mask, normalize)

    def high_pass_filter(self, spec_array: torch.Tensor, normalize: bool = False):
        return self._apply_filter(spec_array, self._hp_mask, normalize)

