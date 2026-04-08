#!/usr/bin/env python3

from typing import Optional, List

import timm
import torch
from torch import nn

from britekit.core.config_loader import get_config
from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class TimmModel(BaseModel):
    """
    Wrapper for models loaded directly from timm.

    When head_type is None, the timm model's built-in head is used and the
    backbone handles classification internally. When head_type is provided,
    the backbone is created without a head (global_pool="", num_classes=0)
    and a custom head from head_factory is attached instead.
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

        cfg = get_config()
        assert model_type.startswith("timm.")

        if head_type is None:
            # Let timm handle classification internally
            self.backbone = timm.create_model(
                model_type[5:],
                pretrained=cfg.train.pretrained,
                in_chans=1,
                num_classes=self.num_classes,
                **kwargs,
            )
            self.head = nn.Identity()
        else:
            # Create backbone as feature extractor, attach custom head
            self.backbone = timm.create_model(
                model_type[5:],
                pretrained=cfg.train.pretrained,
                in_chans=1,
                global_pool="",
                num_classes=0,
                **kwargs,
            )
            # backbone.num_features can be unreliable with global_pool="";
            # use a dummy forward pass to get the actual output channel count
            dummy = torch.zeros(1, 1, cfg.audio.spec_height, cfg.audio.spec_width)
            with torch.no_grad():
                in_channels = self.backbone(dummy).shape[1]
            self.head = make_head(
                head_type,
                in_channels,
                hidden_channels,
                self.num_classes,
                drop_rate=kwargs.pop("drop_rate", 0.0),
            )

    def forward(self, x):
        if self.head_type is None:
            return self.backbone(x), None
        else:
            # Delegate to BaseModel.forward which handles SED vs non-SED routing
            return super().forward(x)
