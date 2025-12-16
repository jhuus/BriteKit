"""
Torch-based spectrogram filtering utilities.

SpecFilter converts a single-channel spectrogram into a 3-channel
representation using low-pass, band-pass, and high-pass frequency masks
applied directly to the spectrogram.
"""

from __future__ import annotations

import torch


class SpecFilter:
    """
    Spectrogram frequency filter operating directly on spectrograms.

    Assumes a fixed spectrogram height defined by cfg.audio.spec_height
    and a fixed device provided at construction time.
    """

    def __init__(self, cfg, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)

        self.H = cfg.audio.spec_height
        self.dtype = torch.float32  # explicit; change if needed

        # Precompute frequency axis
        self._f = torch.linspace(
            0.0, 1.0, self.H, device=self.device, dtype=self.dtype
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
        return torch.sigmoid(
            (self.cfg.audio.low_pass_end - self._f) * self.cfg.audio.low_pass_steepness
        )

    def _create_high_pass_mask(self) -> torch.Tensor:
        return torch.sigmoid(
            (self._f - self.cfg.audio.high_pass_start)
            * self.cfg.audio.high_pass_steepness
        )

    def _create_band_pass_mask(self) -> torch.Tensor:
        low_q = self.cfg.audio.band_pass_start
        high_q = self.cfg.audio.band_pass_end

        if low_q >= high_q:
            raise ValueError("band-pass start must be < end")

        low_edge = torch.sigmoid((self._f - low_q) * self.cfg.audio.band_pass_steepness)
        high_edge = torch.sigmoid(
            (high_q - self._f) * self.cfg.audio.band_pass_steepness
        )

        return low_edge * high_edge

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_2d(spec: torch.Tensor) -> torch.Tensor:
        """
        Ensure spectrogram has shape (H, W).
        Accepts (H, W) or (1, H, W).
        """
        if spec.ndim == 2:
            return spec
        if spec.ndim == 3 and spec.shape[0] == 1:
            return spec[0]
        raise ValueError(
            f"Expected spectrogram of shape (H, W) or (1, H, W), "
            f"got {tuple(spec.shape)}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def low_pass_filter(self, spec: torch.Tensor) -> torch.Tensor:
        spec = self._ensure_2d(spec)
        return spec * self._lp_mask

    def band_pass_filter(self, spec: torch.Tensor) -> torch.Tensor:
        spec = self._ensure_2d(spec)
        return spec * self._bp_mask

    def high_pass_filter(self, spec: torch.Tensor) -> torch.Tensor:
        spec = self._ensure_2d(spec)
        return spec * self._hp_mask

    def filter(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Convert a 1-channel spectrogram into a 3-channel spectrogram.

        Input:
            spec: (H, W) or (1, H, W)
            (must already be on self.device)

        Output:
            (3, H, W) -> [low-pass, band-pass, high-pass]
        """
        spec = self._ensure_2d(spec)

        return torch.stack(
            (
                spec * self._lp_mask,
                spec * self._bp_mask,
                spec * self._hp_mask,
            ),
            dim=0,
        )
