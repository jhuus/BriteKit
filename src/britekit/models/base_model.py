#!/usr/bin/env python3

import contextlib
from datetime import datetime
from typing import List, Optional, Any
import uuid

import numpy as np
import pytorch_lightning as pl
from sklearn import metrics
from timm.optim import create_optimizer_v2
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from britekit.core.config_loader import get_config
from britekit.core import util
from britekit.models.head_factory import is_sed


def get_learning_rate(optimizer):
    """Get learning rates from all parameter groups."""
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group["lr"])
    return lrs[0] if len(lrs) == 1 else lrs


class BaseModel(pl.LightningModule):
    """
    Base class for models.

    Attributes:
        model_type (str): e.g. "effnet.5" or "timm.efficientnet_s"
        head_type (str, optional): None for default head, else "basic", "basic_sed" etc.
        hidden_channels (int): Used in scaled models if head specified, for some head types.
        train_class_names (list[str]): List of training class names
        train_class_codes (list[str]): List of training class codes
        train_class_alt_names (list[str]): List of training class alternate names
        train_class_alt_codes (list[str]): List of training class alternate codes
        num_train_specs (int): Number of training spectrograms
        multi_label (bool): If true, train multi_label model, else multi_class model.
    """

    # ==================================================================
    # Initialization & configuration
    # ==================================================================

    def __init__(
        self,
        model_type: str,
        head_type: Optional[str],
        hidden_channels: int,
        train_class_names: List[str],
        train_class_codes: List[str],
        train_class_alt_names: List[str],
        train_class_alt_codes: List[str],
        num_train_specs: int,
        multi_label: bool,
    ):
        super().__init__()

        # Input validation
        if not train_class_names:
            raise ValueError("train_class_names cannot be empty")
        if len(train_class_names) != len(train_class_codes):
            raise ValueError(
                "train_class_names and train_class_codes must have the same length"
            )
        if len(train_class_names) != len(train_class_alt_names):
            raise ValueError(
                "train_class_names and train_class_alt_names must have the same length"
            )
        if len(train_class_names) != len(train_class_alt_codes):
            raise ValueError(
                "train_class_names and train_class_alt_codes must have the same length"
            )

        self.save_hyperparameters()
        self.cfg = get_config()

        # Save parameters
        self.model_type = model_type
        self.head_type = head_type
        self.use_sed = is_sed(head_type)
        self.hidden_channels = hidden_channels
        self.multi_label = multi_label
        self.train_class_names = train_class_names
        self.train_class_codes = train_class_codes
        self.train_class_alt_names = train_class_alt_names
        self.train_class_alt_codes = train_class_alt_codes
        self.num_train_specs = num_train_specs
        self.num_classes = len(train_class_names)
        self.learning_rate = self.cfg.train.learning_rate

        # Loss function
        if self.multi_label:
            self.loss_fn: Any = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Model components (defined by subclass)
        self.backbone: Optional[nn.Module] = None
        self.head: Optional[nn.Module] = None

    # ==================================================================
    # Lightning lifecycle hooks
    # ==================================================================

    def on_save_checkpoint(self, checkpoint):
        if not hasattr(self, "identifier"):
            self.identifier = str(uuid.uuid4()).upper()
            self.training_date = datetime.today().strftime("%Y-%m-%d")

        checkpoint["identifier"] = self.identifier
        checkpoint["training_date"] = self.training_date
        checkpoint["training_cfg"] = util.cfg_to_pure(self.cfg)

    def on_load_checkpoint(self, checkpoint):
        if "identifier" in checkpoint:
            self.identifier = checkpoint["identifier"]
            self.training_date = checkpoint["training_date"]
            self.training_cfg = checkpoint["training_cfg"]

            self.cfg.train.sed_fps = self.training_cfg["train"]["sed_fps"]
            self.cfg.audio.spec_duration = self.training_cfg["audio"]["spec_duration"]
        else:
            raise ValueError("Checkpoint metadata not found.")

    # ==================================================================
    # Forward pass
    # ==================================================================

    def forward(self, x):
        if self.backbone is None:
            raise RuntimeError("Backbone is not initialized.")
        if self.head is None:
            raise RuntimeError("Head is not initialized.")

        x = self.backbone(x)
        x = self.head(x)

        if self.use_sed:
            segment_logits, frame_logits = x
            actual = frame_logits.shape[-1]
            target = int(self.cfg.train.sed_fps * self.cfg.audio.spec_duration)
            if actual != target:
                frame_logits = F.interpolate(frame_logits, size=target, mode="linear")
            return segment_logits, frame_logits
        else:
            return x, None

    # ==================================================================
    # Training / validation / testing
    # ==================================================================

    def training_step(self, batch, batch_idx):
        input = batch["input"]
        seg_labels = batch["segment_labels"]

        if self.multi_label:
            seg_labels = (
                seg_labels * (1.0 - self.cfg.train.pos_label_smoothing)
                + (1.0 - seg_labels) * self.cfg.train.neg_label_smoothing
            )

        seg_logits, frame_logits = self(input)
        loss = self._calc_loss(seg_logits, frame_logits, seg_labels)

        if frame_logits is not None and self.cfg.train.offpeak_weight > 0:
            p = torch.sigmoid(frame_logits)
            mask = ~batch["mixup"].bool()
            m = mask.view(-1, 1, 1).float()
            p_sum = (p * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
            p_max = (p * m).amax(dim=-1)
            loss += self.cfg.train.offpeak_weight * (p_sum - p_max).clamp_min(0).mean()

        self.log(
            "lr",
            get_learning_rate(self.optimizer),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["segment_labels"]
        seg_logits, _ = self(x)
        loss = self.loss_fn(seg_logits, y)

        preds = (
            torch.sigmoid(seg_logits)
            if self.multi_label
            else torch.softmax(seg_logits, dim=1)
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val_roc",
            metrics.roc_auc_score(y.cpu(), preds.cpu(), average="micro"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        seg_logits, _ = self(x)
        loss = self.loss_fn(seg_logits, y)
        self.log("test_loss", loss, on_epoch=True)

        if self.multi_label:
            preds = torch.sigmoid(seg_logits)
            self.log(
                "test_roc_auc",
                metrics.roc_auc_score(y.cpu(), preds.cpu(), average="micro"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    # ==================================================================
    # Optimizers & schedulers
    # ==================================================================

    def configure_optimizers(self):
        cfg = self.cfg.train
        self.optimizer = create_optimizer_v2(
            self,
            cfg.optimizer,
            lr=cfg.learning_rate,
            weight_decay=cfg.opt_weight_decay,
            betas=(cfg.opt_beta1, cfg.opt_beta2),
            filter_bias_and_bn=False,
        )

        total_steps = (
            self.trainer.estimated_stepping_batches
            if hasattr(self, "trainer") and self.trainer
            else 1000
        )

        warmup_steps = self.cfg.train.warmup_fraction * total_steps
        decay_steps = total_steps - warmup_steps

        cosine = CosineAnnealingLR(self.optimizer, T_max=decay_steps)
        if warmup_steps > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = SequentialLR(
                self.optimizer, [warmup, cosine], milestones=[warmup_steps]
            )
        else:
            scheduler = cosine

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ==================================================================
    # Inference & embeddings
    # ==================================================================

    def predict(self, x, device=None):
        """
        Memory-safe, block-wise inference with a single AMP toggle.
        Config:
        - infer.block_size: int (None/0 -> whole batch)
        - infer.autocast: bool (use CUDA autocast if on GPU)
        - infer.scaling_coefficient / scaling_intercept: scalar or [C] (multi-label only)
        Returns:
        segment_scores: (N, C) cpu float32 tensor
        frame_scores:   (N, C, T) cpu float32 tensor or None
        """
        if device is None:
            device = util.get_device()

        block_size = self.cfg.infer.block_size
        # Fix device handling
        use_amp = bool(self.cfg.infer.autocast and device and device.startswith("cuda"))

        # choose a safe AMP dtype (bf16 if available; else fp16)
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        amp_ctx = (
            (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype))
            if use_amp
            else (lambda: contextlib.nullcontext())
        )

        # Move _iter_blocks outside to avoid redefinition
        seg_parts, frame_parts = [], []

        with torch.inference_mode():
            for x_block in self._iter_blocks(x, block_size):
                xb = self._ensure_tensor(x_block, device)

                # Heavy ops under autocast (if enabled)
                with amp_ctx():
                    seg_logits, frame_logits = self(xb)

                # === back to fp32 for numerics & calibration ===
                seg_logits = seg_logits.float()
                frame_logits = None if frame_logits is None else frame_logits.float()

                # segment scores
                if self.multi_label:
                    w = torch.as_tensor(
                        self.cfg.infer.scaling_coefficient,
                        device=seg_logits.device,
                        dtype=seg_logits.dtype,
                    )
                    b = torch.as_tensor(
                        self.cfg.infer.scaling_intercept,
                        device=seg_logits.device,
                        dtype=seg_logits.dtype,
                    )
                    seg_scores = torch.sigmoid(seg_logits * w + b)
                else:
                    seg_scores = torch.softmax(seg_logits, dim=1)

                seg_parts.append(seg_scores.detach().cpu())

                # frame scores (SED)
                if frame_logits is not None:
                    frame_scores = torch.sigmoid(frame_logits)
                    frame_parts.append(frame_scores.detach().cpu())

        segment_scores = torch.cat(seg_parts, dim=0)
        frame_scores = torch.cat(frame_parts, dim=0) if frame_parts else None
        return segment_scores, frame_scores

    def get_embeddings(self, specs, device=None):
        """Get embeddings for use in searching and clustering"""
        if device is None:
            device = util.get_device()

        with torch.no_grad():
            specs = self._ensure_tensor(specs, device)
            feats = self.backbone(specs)

            # If already 2D (e.g., [B, D]), just return
            if feats.ndim == 2:
                return feats.cpu().detach().numpy()

            # 3D (SED) or 4D (CNN feature map)
            if feats.ndim == 3:  # [B, C, T]
                pooled = feats.mean(dim=-1)  # global temporal pooling
            elif feats.ndim == 4:  # [B, C, H, W]
                pooled = (
                    F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
                )  # [B, C]
            else:
                raise ValueError(f"Unexpected feature shape: {feats.shape}")

            return pooled.cpu().detach().numpy()  # [B, D]

    def get_sed_scores(self, specs, device=None):
        """
        Get segment-scores and frame-scores from a model with a SED head.
        """
        assert self.use_sed, "get_sed_scores called on a non-SED model"
        if device is None:
            device = util.get_device()

        self.backbone.to(device)
        self.backbone.eval()
        self.head.to(device)
        self.head.eval()

        with torch.no_grad():
            specs = self._ensure_tensor(specs, device)
            segment_logits, frame_logits = self(specs)
            segment_scores = 1 / (1 + torch.exp(-segment_logits))
            frame_scores = torch.sigmoid(frame_logits)

        segment_scores = segment_scores.cpu().detach().numpy()
        frame_scores = frame_scores.cpu().detach().numpy()
        return segment_scores, frame_scores

    # ==================================================================
    # Utilities & helpers
    # ==================================================================

    def freeze_backbone(self):
        if self.backbone:
            for _, p in self.backbone.named_parameters():
                p.requires_grad = False

    def _calc_loss(self, seg_logits, frame_logits, seg_labels):
        segment_loss = self.loss_fn(seg_logits, seg_labels)

        if self.use_sed:
            assert frame_logits is not None
            B, C, T = frame_logits.shape
            frame_labels = seg_labels.unsqueeze(-1).expand(B, C, T).transpose(1, 2)
            frame_loss = F.binary_cross_entropy_with_logits(
                frame_logits.transpose(1, 2), frame_labels, reduction="mean"
            )
            segment_loss_weight = 1 - self.cfg.train.frame_loss_weight
            loss = (
                segment_loss_weight * segment_loss
                + self.cfg.train.frame_loss_weight * frame_loss
            )
        else:
            loss = segment_loss

        return loss

    def _ensure_tensor(self, x, device=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not torch.is_tensor(x):
            raise TypeError(f"Expected tensor or ndarray, got {type(x)}")

        x = x.to(dtype=torch.float32)
        if device is not None:
            x = x.to(device)
        return x

    def _iter_blocks(self, X, block_size):
        """Helper method to iterate over data in blocks."""
        if isinstance(X, torch.Tensor):
            n = X.shape[0]
            if block_size <= 0 or block_size >= n:
                yield X
            else:
                for i in range(0, n, block_size):
                    yield X[i : i + block_size]
        else:
            n = len(X)
            if block_size <= 0 or block_size >= n:
                yield X
            else:
                for i in range(0, n, block_size):
                    yield X[i : i + block_size]
