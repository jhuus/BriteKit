#!/usr/bin/env python3

from typing import List, Optional

from timm.models import mobilenetv3
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class MobileNet(BaseModel):
    """Width-scaled MobileNetV4 Hybrid Medium models."""

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

        channel_multiplier = MODEL_REGISTRY[model_type]

        def make_backbone(**backbone_kwargs):
            return _gen_mobilenet(channel_multiplier, **backbone_kwargs)

        model_kwargs = dict(in_chans=1, **kwargs)

        if head_type is None:
            self.backbone = make_backbone(
                num_classes=self.num_classes,
                **model_kwargs,
            )
            self.head = nn.Identity()
        else:
            # This is the same feature-extractor boundary used by TimmModel:
            # retain the final 1280-channel projection, but remove pooling and
            # the class-dependent linear classifier.
            self.backbone = make_backbone(
                global_pool="",
                num_classes=0,
                **model_kwargs,
            )
            self.head = make_head(
                head_type,
                self.backbone.head_hidden_size,
                hidden_channels,
                self.num_classes,
                drop_rate=kwargs.get("drop_rate", 0.0),
                lse_temp=kwargs.get("lse_temp", 0.5),
                two_way=kwargs.get("two_way", True),
            )


# Channel multipliers for the timm MobileNetV4 Hybrid Medium architecture.
# Counts exclude the class-dependent classifier and use a one-channel input.
MODEL_REGISTRY = {
    "mobilenet.1": 0.26,  # ~0.97M parameters
    "mobilenet.2": 0.42,  # ~1.98M parameters
    "mobilenet.3": 0.52,  # ~2.97M parameters
    "mobilenet.4": 0.63,  # ~4.06M parameters (tested 4s3x2 architecture)
    "mobilenet.5": 0.71,  # ~5.00M parameters
    "mobilenet.6": 0.79,  # ~5.92M parameters
}


def _gen_mobilenet(channel_multiplier: float, **kwargs):
    """MobileNetV4 Hybrid Medium retaining its final two attention blocks."""
    model = mobilenetv3._gen_mobilenet_v4(
        "mobilenetv4_hybrid_medium",
        channel_multiplier=channel_multiplier,
        **kwargs,
    )
    model.blocks[2] = nn.Sequential(
        *(
            block
            for block in model.blocks[2]
            if block.__class__.__name__ != "MobileAttention"
        )
    )
    attention_seen = 0
    retained_blocks = []
    for block in reversed(model.blocks[3]):
        if block.__class__.__name__ == "MobileAttention":
            attention_seen += 1
            if attention_seen > 2:
                continue
        retained_blocks.append(block)
    model.blocks[3] = nn.Sequential(*reversed(retained_blocks))
    return model
