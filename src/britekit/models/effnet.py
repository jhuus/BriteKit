#!/usr/bin/env python3

import copy
from typing import cast, List, Optional

from timm.models import efficientnet
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class EffNetModel(BaseModel):
    """
    Scaled version of timm EfficientNet V2, where model_size parameter defines the scaling.
    """

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

        channel_multiplier, depth_multiplier = MODEL_REGISTRY[model_type]

        self.backbone = efficientnet._gen_efficientnetv2_s(
            "efficientnetv2_rw_t",
            channel_multiplier=channel_multiplier,
            depth_multiplier=depth_multiplier,
            num_classes=self.num_classes,
            in_chans=1,
            **kwargs,
        )

        if head_type is None:
            self.head = nn.Sequential(
                cast(nn.Module, copy.deepcopy(self.backbone.global_pool)),
                cast(nn.Module, copy.deepcopy(self.backbone.classifier)),
            )
        else:
            in_channels = cast(int, self.backbone.num_features)
            self.head = make_head(
                head_type,
                in_channels,
                hidden_channels,
                self.num_classes,
                drop_rate=kwargs.pop("drop_rate", 0.0),
            )

        # remove the old head, preserving the Conv2d, BatchNorm2d and
        # SiLU layers, which are used by forward_features
        self.backbone.global_pool = nn.Identity()
        self.backbone.classifier = nn.Identity()


# (channel_multiplier, depth_multiplier) per model type
MODEL_REGISTRY = {
    "effnet.1":
    # ~400K parameters with 50 classes
    (0.25, 0.26),
    "effnet.2":
    # ~690K parameters with 50 classes
    (0.3, 0.3),
    "effnet.3":
    # ~1.4M parameters with 50 classes
    (0.4, 0.4),
    "effnet.4":
    # ~2M parameters with 50 classes
    (0.42, 0.51),
    "effnet.5":
    # ~2.9M parameters with 50 classes
    (0.48, 0.58),
    "effnet.6":
    # ~3.7M parameters with 50 classes
    (0.54, 0.6),
    "effnet.7":
    # ~4.6M parameters with 50 classes
    (0.6, 0.6),
    "effnet.8":
    # ~5.4M parameters with 50 classes
    (0.65, 0.6),
    "effnet.9":
    # ~6.1M parameters with 50 classes
    (0.64, 0.7),
    "effnet.10":
    # ~7.3M parameters with 50 classes
    (0.7, 0.7),
    "effnet.11":
    # ~8.1M parameters with 50 classes
    (0.7, 0.8),
}
