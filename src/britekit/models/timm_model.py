#!/usr/bin/env python3

from typing import Optional, List

import timm
from torch import nn

from britekit.core.config_loader import get_config
from britekit.models.base_model import BaseModel


class TimmModel(BaseModel):
    """
    Wrapper for models loaded directly from timm.
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

        # head replacement is not supported here since it
        # would be very complicated with so many model types
        cfg = get_config()
        assert model_type.startswith("timm.")

        self.backbone = timm.create_model(
            model_type[5:],  # strip off the "timm." prefix
            pretrained=cfg.train.pretrained,
            in_chans=1,
            num_classes=self.num_classes,
            **kwargs,
        )
        self.head = nn.Identity()

    def forward(self, x):
        return self.backbone(x), None
