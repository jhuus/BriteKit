#!/usr/bin/env python3

import copy
from typing import List, Optional

from timm.models import dla
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class DlaModel(BaseModel):
    """
    Scaled version of timm DLA, where model_size parameter defines the scaling.
    DLA Paper: `Deep Layer Aggregation` - https://arxiv.org/abs/1707.06484.
    This is a very slow model, so only small sizes are provided.
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

        config = MODEL_REGISTRY[model_type]

        # the "type" comment below is an instruction to MyPy
        self.backbone = dla.DLA(in_chans=1, num_classes=self.num_classes, **config, **kwargs)  # type: ignore

        if head_type is None:
            self.head = nn.Sequential(
                copy.deepcopy(self.backbone.global_pool),
                copy.deepcopy(self.backbone.head_drop),
                copy.deepcopy(self.backbone.fc),
                copy.deepcopy(self.backbone.flatten),
            )
        else:
            in_channels = self.backbone.num_features
            self.head = make_head(
                head_type,
                in_channels,
                hidden_channels,
                self.num_classes,
                drop_rate=kwargs.pop("drop_rate", 0.0),
            )

        self.backbone.global_pool = nn.Identity()
        self.backbone.head_drop = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.backbone.flatten = nn.Identity()


# Model size is most affected by number of classes for smaller models
MODEL_REGISTRY = {
    "dla.1":
    # ~390K parameters with 50 classes
    dict(
        levels=[1, 1, 1, 1, 1, 1],
        channels=[16, 32, 32, 32, 64, 64],
        block=dla.DlaBasic,
    ),
    "dla.2":
    # ~630K parameters with 50 classes
    dict(
        levels=[1, 1, 1, 1, 1, 1],
        channels=[16, 32, 64, 64, 64, 64],
        block=dla.DlaBasic,
    ),
    "dla.3":
    # ~1.0M parameters with 50 classes
    dict(
        levels=[1, 1, 1, 1, 1, 1],
        channels=[16, 32, 64, 64, 64, 128],
        block=dla.DlaBasic,
    ),
    "dla.4":
    # ~1.5M parameters with 50 classes
    dict(
        levels=[1, 1, 1, 1, 1, 1],
        channels=[16, 32, 64, 64, 128, 128],
        block=dla.DlaBasic,
    ),
    "dla.5":
    # ~2.1M parameters
    dict(
        levels=[1, 1, 1, 1, 1, 1],
        channels=[16, 32, 64, 96, 128, 160],
        block=dla.DlaBasic,
    ),
}
