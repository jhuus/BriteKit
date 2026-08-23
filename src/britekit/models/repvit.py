#!/usr/bin/env python3

from typing import List, Optional

from timm.models import repvit
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class RepVitModel(BaseModel):
    """Scaled timm RepViT models."""

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
        **kwargs,
    ):
        super().__init__(
            model_type,
            head_type,
            hidden_channels,
            train_class_names,
            train_class_codes,
            train_class_alt_names,
            train_class_alt_codes,
            num_train_specs,
            multi_label,
        )

        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        # RepViT does not implement stochastic depth.
        drop_path_rate = kwargs.pop("drop_path_rate", None)
        if drop_path_rate not in (None, 0.0):
            raise ValueError("RepViT does not support drop_path_rate")

        embed_dim, depth = MODEL_REGISTRY[model_type]
        model_kwargs = dict(
            embed_dim=embed_dim,
            depth=depth,
            in_chans=1,
            **kwargs,
        )
        if head_type is None:
            self.backbone = repvit.RepVit(
                num_classes=self.num_classes,
                **model_kwargs,
            )
            self.head = nn.Identity()
        else:
            self.backbone = repvit.RepVit(
                global_pool="",
                num_classes=0,
                **model_kwargs,
            )
            self.head = make_head(
                head_type,
                self.backbone.num_features,
                hidden_channels,
                self.num_classes,
                drop_rate=kwargs.get("drop_rate", 0.0),
                lse_temp=kwargs.get("lse_temp", 0.5),
                two_way=kwargs.get("two_way", True),
            )


# ((stage widths), (stage depths)) per model type. Repvit.5 exactly matches
# timm.repvit_m0_9. Counts exclude the class-dependent classifier and use a
# one-channel input.
MODEL_REGISTRY = {
    "repvit.1": ((24, 48, 96, 192), (2, 2, 10, 2)),  # ~1.04M parameters
    "repvit.2": ((32, 64, 128, 256), (2, 2, 12, 2)),  # ~1.97M parameters
    "repvit.3": ((40, 80, 160, 320), (2, 2, 12, 2)),  # ~3.07M parameters
    "repvit.4": ((40, 80, 160, 320), (2, 2, 20, 2)),  # ~3.96M parameters
    "repvit.5": ((48, 96, 192, 384), (2, 2, 14, 2)),  # ~4.72M; timm M0.9
    "repvit.6": ((48, 96, 192, 384), (2, 2, 16, 2)),  # ~5.04M parameters
    "repvit.7": ((48, 96, 192, 384), (2, 2, 22, 2)),  # ~6.00M parameters
}
