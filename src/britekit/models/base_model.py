#!/usr/bin/env python3

import contextlib
from datetime import datetime
import logging
from typing import List, Optional, Any
import uuid

import numpy as np
import lightning.pytorch as pl
from timm.optim import create_optimizer_v2
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from britekit import __version__ as britekit_version
from britekit.core.config_loader import get_config, BaseConfig
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
        self._val_preds: List = []
        self._val_labels: List = []

        # Loss function
        if self.multi_label:
            self.loss_fn: Any = nn.BCEWithLogitsLoss(
                weight=torch.ones(self.num_classes)
            )
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
        training_cfg: Any = util.cfg_to_pure(self.cfg)
        # Lightning stores a zero-based epoch in each checkpoint.  Record the
        # number of epochs actually completed by this checkpoint rather than
        # the configured maximum so downstream manifests describe the model
        # that was loaded (which may be an earlier retained checkpoint).
        training_cfg["train"]["num_epochs"] = checkpoint["epoch"] + 1
        checkpoint["training_cfg"] = training_cfg
        checkpoint["britekit_version"] = britekit_version

    def on_load_checkpoint(self, checkpoint):
        if "identifier" in checkpoint:
            self.identifier = checkpoint["identifier"]
            self.training_date = checkpoint["training_date"]
            self.training_cfg = checkpoint["training_cfg"]

            # Checkpoints created before num_epochs represented the configured
            # maximum still contain Lightning's actual zero-based epoch.
            if "epoch" in checkpoint:
                self.training_cfg["train"]["num_epochs"] = checkpoint["epoch"] + 1

            self.cfg.audio.spec_duration = self.training_cfg["audio"]["spec_duration"]
            self.cfg.audio.spec_height = self.training_cfg["audio"]["spec_height"]
            self.cfg.audio.spec_width = self.training_cfg["audio"]["spec_width"]
            self.cfg.audio.win_length = self.training_cfg["audio"]["win_length"]
            self.cfg.audio.max_freq = self.training_cfg["audio"]["max_freq"]
            self.cfg.audio.min_freq = self.training_cfg["audio"]["min_freq"]
            self.cfg.audio.sampling_rate = self.training_cfg["audio"]["sampling_rate"]
            self.cfg.audio.freq_scale = self.training_cfg["audio"]["freq_scale"]
            self.cfg.audio.power = self.training_cfg["audio"]["power"]
            self.cfg.audio.decibels = self.training_cfg["audio"]["decibels"]
            self.cfg.audio.top_db = self.training_cfg["audio"].get(
                "top_db", self.cfg.audio.top_db
            )
            self.cfg.audio.db_power = self.training_cfg["audio"].get(
                "db_power", self.cfg.audio.db_power
            )
            self.cfg.audio.log_freq_gain = self.training_cfg["audio"].get(
                "log_freq_gain", self.cfg.audio.log_freq_gain
            )
            self.cfg.audio.mel_norm = self.training_cfg["audio"].get(
                "mel_norm", self.cfg.audio.mel_norm
            )

            self.cfg.train.sed_fps = self.training_cfg["train"]["sed_fps"]
            self.cfg.train.model_type = self.training_cfg["train"]["model_type"]
            self.cfg.train.head_type = self.training_cfg["train"].get("head_type")
            self.cfg.train.lse_temp = self.training_cfg["train"].get("lse_temp", 0.5)
            self.cfg.train.two_way = self.training_cfg["train"].get("two_way", True)

            if "n_fft" in self.training_cfg["audio"]:
                self.cfg.audio.n_fft = self.training_cfg["audio"]["n_fft"]
            else:
                win_length_samples = int(
                    self.cfg.audio.win_length * self.cfg.audio.sampling_rate
                )
                self.cfg.audio.n_fft = 2 * win_length_samples

            logging.debug(
                "BaseModel::on_load_checkpoint sr=%d, win=%d, duration=%.2f, height=%d, width=%d",
                self.cfg.audio.sampling_rate,
                self.cfg.audio.win_length,
                self.cfg.audio.spec_duration,
                self.cfg.audio.spec_height,
                self.cfg.audio.spec_width,
            )
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
            target = int(self.cfg.train.sed_fps * self.cfg.audio.spec_duration)
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
        raw_labels = batch["segment_labels"]
        frame_labels = batch.get("frame_labels")  # (B, 12, num_classes) or None
        teacher_labels = batch.get("teacher_segment_labels")
        teacher_frame_labels = batch.get("teacher_frame_labels")
        hard_segment_mask = batch.get("hard_segment_mask")
        frame_label_mask = batch.get("frame_label_mask")

        if self.multi_label:
            seg_labels = (
                seg_labels * (1.0 - self.cfg.train.pos_label_smoothing)
                + (1.0 - seg_labels) * self.cfg.train.neg_label_smoothing
            )
            if frame_labels is not None:
                frame_labels = (
                    frame_labels * (1.0 - self.cfg.train.pos_label_smoothing)
                    + (1.0 - frame_labels) * self.cfg.train.neg_label_smoothing
                )

        seg_logits, frame_logits = self(input)
        loss = self._calc_loss(
            seg_logits,
            frame_logits,
            seg_labels,
            raw_labels,
            frame_labels,
            teacher_labels,
            teacher_frame_labels,
            hard_segment_mask,
            frame_label_mask,
        )

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

        self._val_preds.append(preds.cpu())
        self._val_labels.append(y.cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        from sklearn import metrics

        if not self._val_preds:
            return
        all_preds = torch.cat(self._val_preds).numpy()
        all_labels = (torch.cat(self._val_labels) >= 0.5).int().numpy()
        self._val_preds.clear()
        self._val_labels.clear()
        self.log(
            "val_roc",
            metrics.roc_auc_score(all_labels, all_preds, average="micro"),
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        from sklearn import metrics

        x, y = batch
        seg_logits, _ = self(x)
        loss = self.loss_fn(seg_logits, y)
        self.log("test_loss", loss, on_epoch=True)

        if self.multi_label:
            preds = torch.sigmoid(seg_logits)
            self.log(
                "test_roc_auc",
                metrics.roc_auc_score(
                    (y.cpu() >= 0.5).int(), preds.cpu(), average="micro"
                ),
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

        kwargs = {
            "lr": cfg.learning_rate,
            "filter_bias_and_bn": False,
        }
        if cfg.opt_weight_decay is not None:
            kwargs["weight_decay"] = cfg.opt_weight_decay
        if cfg.opt_beta1 is not None:
            kwargs["betas"] = (cfg.opt_beta1, cfg.opt_beta2)

        self.optimizer = create_optimizer_v2(self, cfg.optimizer, **kwargs)

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

        logging.debug(
            "BaseModel::predict multi_label=%s, scaling_coefficient=%.3f, scaling_intercept=%.3f",
            self.multi_label,
            self.cfg.infer.scaling_coefficient,
            self.cfg.infer.scaling_intercept,
        )
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

                seg_parts.append(seg_scores)

                # frame scores (SED)
                if frame_logits is not None:
                    if self.multi_label:
                        frame_scores = torch.sigmoid(frame_logits * w + b)
                    else:
                        frame_scores = torch.softmax(frame_logits, dim=1)

                    frame_parts.append(frame_scores)

        segment_scores = torch.cat(seg_parts, dim=0)
        frame_scores = torch.cat(frame_parts, dim=0) if frame_parts else None

        if frame_scores is None:
            return segment_scores.cpu().numpy(), None
        else:
            return segment_scores.cpu().numpy(), frame_scores.cpu().numpy()

    def get_embeddings(self, specs, device=None):
        """Get embeddings for use in searching and clustering"""
        if device is None:
            device = util.get_device()

        with torch.no_grad():
            specs = self._ensure_tensor(specs, device)
            feats = self.backbone(specs)

            # If already 2D (e.g., [B, D]), just return
            if feats.ndim == 2:
                return feats.cpu().numpy()

            # 3D (SED) or 4D (CNN feature map)
            if feats.ndim == 3:  # [B, C, T]
                pooled = feats.mean(dim=-1)  # global temporal pooling
            elif feats.ndim == 4:  # [B, C, H, W]
                pooled = (
                    F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
                )  # [B, C]
            else:
                raise ValueError(f"Unexpected feature shape: {feats.shape}")

            return pooled.cpu().numpy()  # [B, D]

    # ==================================================================
    # Utilities & helpers
    # ==================================================================

    def freeze_backbone(self):
        if self.backbone:
            for _, p in self.backbone.named_parameters():
                p.requires_grad = False

    def set_class_weights(self, class_weights):
        """Set class weights on the loss function."""
        import torch

        if class_weights is None:
            return

        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        if self.multi_label:
            self.loss_fn = nn.BCEWithLogitsLoss(weight=weights_tensor)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

    def set_config(self, cfg: BaseConfig):
        self.cfg = cfg

    def _calc_loss(
        self,
        seg_logits,
        frame_logits,
        seg_labels,
        raw_labels,
        frame_labels=None,
        teacher_labels=None,
        teacher_frame_labels=None,
        hard_segment_mask=None,
        frame_label_mask=None,
    ):
        segment_losses = self._loss_per_sample(seg_logits, seg_labels)

        if teacher_labels is not None:
            weight = self.cfg.train.distillation_weight
            temperature = self.cfg.train.distillation_temperature
            if not 0 <= weight <= 1:
                raise ValueError("distillation_weight must be between 0 and 1")
            if temperature <= 0:
                raise ValueError("distillation_temperature must be greater than 0")
            softened_targets = self._soften_targets(teacher_labels, temperature)
            teacher_losses = self._loss_per_sample(
                seg_logits / temperature, softened_targets
            )
            teacher_losses *= temperature**2
            blended_losses = (1 - weight) * segment_losses + weight * teacher_losses
            if hard_segment_mask is not None:
                segment_losses = torch.where(
                    hard_segment_mask.bool(), blended_losses, teacher_losses
                )
            else:
                segment_losses = blended_losses
        elif hard_segment_mask is not None and not hard_segment_mask.bool().all():
            raise ValueError("Teacher-only samples require teacher segment labels")

        if self.use_sed:
            assert frame_logits is not None
            B, C, T = frame_logits.shape
            if frame_labels is not None:
                # Upsample stored frame labels from 12 → T frames
                # frame_labels: (B, 12, C) → permute → (B, C, 12) → interpolate → (B, C, T) → (B, T, C)
                fl = frame_labels.permute(0, 2, 1)  # (B, C, 12)
                fl = F.interpolate(fl, size=T, mode="nearest")  # (B, C, T)
                frame_labels = fl.permute(0, 2, 1)  # (B, T, C)
            else:
                frame_labels = seg_labels.unsqueeze(-1).expand(B, C, T).transpose(1, 2)
            frame_losses = F.binary_cross_entropy_with_logits(
                frame_logits.transpose(1, 2), frame_labels, reduction="none"
            )
            frame_losses = frame_losses.mean(dim=(1, 2))

            if teacher_frame_labels is not None:
                teacher_frames = teacher_frame_labels.permute(0, 2, 1)
                teacher_frames = F.interpolate(
                    teacher_frames, size=T, mode="linear", align_corners=False
                ).permute(0, 2, 1)
                temperature = self.cfg.train.distillation_temperature
                softened_frames = self._soften_targets(teacher_frames, temperature)
                teacher_frame_losses = F.binary_cross_entropy_with_logits(
                    frame_logits.transpose(1, 2) / temperature,
                    softened_frames,
                    reduction="none",
                ).mean(dim=(1, 2))
                teacher_frame_losses *= temperature**2
                if frame_label_mask is not None:
                    frame_losses = torch.where(
                        frame_label_mask.bool(), frame_losses, teacher_frame_losses
                    )
            elif frame_label_mask is not None and not frame_label_mask.bool().all():
                raise ValueError("Teacher-only samples require teacher frame labels")

            frame_weight = self.cfg.train.frame_loss_weight
            sample_losses = (
                1 - frame_weight
            ) * segment_losses + frame_weight * frame_losses
            loss = sample_losses.mean()
        else:
            loss = segment_losses.mean()

        return loss

    @staticmethod
    def _soften_targets(targets, temperature):
        eps = torch.finfo(targets.dtype).eps
        target_logits = torch.logit(targets.clamp(eps, 1 - eps))
        return torch.sigmoid(target_logits / temperature)

    def _loss_per_sample(self, logits, labels):
        """Calculate classification loss while retaining the batch dimension."""
        if self.multi_label:
            losses = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            if self.loss_fn.weight is not None:
                losses = losses * self.loss_fn.weight.to(losses.device)
            return losses.mean(dim=1)

        return F.cross_entropy(
            logits,
            labels,
            weight=self.loss_fn.weight,
            reduction="none",
        )

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
