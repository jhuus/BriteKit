#!/usr/bin/env python3

import copy
from typing import cast, List, Optional

from timm.layers import ClassifierHead
from timm.models import nfnet
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class NfNetModel(BaseModel):
    """
    Scaled version of timm NFNet, where model_size parameter defines the scaling.
    Papers:
      `Characterizing signal propagation to close the performance gap in unnormalized ResNets` - https://arxiv.org/abs/2101.08692
      `High-Performance Large-Scale Image Recognition Without Normalization` - https://arxiv.org/abs/2102.06171
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
        self.backbone = nfnet.NormFreeNet(
            cfg=config, num_classes=self.num_classes, in_chans=1, **kwargs
        )

        if head_type is None:
            self.head = copy.deepcopy(self.backbone.head)
        else:
            in_channels = self.backbone.num_features
            self.head = make_head(
                head_type,
                in_channels,
                hidden_channels,
                self.num_classes,
                drop_rate=kwargs.pop("drop_rate", 0.0),
                lse_temp=kwargs.pop("lse_temp", 0.5),
                two_way=kwargs.pop("two_way", True),
            )

        self.backbone.head = cast(ClassifierHead, nn.Identity())


# Initial custom configurations for size checks before expanding the set.
MODEL_REGISTRY = {
    "nfnet.1":
    # ~125K parameters
    nfnet.NfCfg(
        depths=(1, 1, 2, 1),
        channels=(40, 80, 120, 160),
        stem_type="3x3",
        stem_chs=32,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=240,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
    "nfnet.2":
    # ~220K parameters
    nfnet.NfCfg(
        depths=(1, 2, 2, 1),
        channels=(48, 96, 160, 224),
        stem_type="3x3",
        stem_chs=32,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=336,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
    "nfnet.3":
    # Target: ~500K parameters
    nfnet.NfCfg(
        depths=(1, 2, 3, 1),
        channels=(64, 128, 224, 320),
        stem_type="3x3",
        stem_chs=40,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=480,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
    "nfnet.4":
    # ~930K parameters
    nfnet.NfCfg(
        depths=(1, 2, 4, 1),
        channels=(88, 176, 320, 464),
        stem_type="3x3",
        stem_chs=56,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=640,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
    "nfnet.5":
    # Target: ~2M parameters
    nfnet.NfCfg(
        depths=(1, 3, 5, 2),
        channels=(96, 192, 384, 576),
        stem_type="3x3",
        stem_chs=56,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=768,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
    "nfnet.6":
    # ~2.5M parameters
    nfnet.NfCfg(
        depths=(1, 3, 6, 2),
        channels=(120, 240, 464, 688),
        stem_type="3x3",
        stem_chs=72,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=896,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
    "nfnet.7":
    # ~3.3M parameters
    nfnet.NfCfg(
        depths=(1, 3, 7, 2),
        channels=(136, 272, 528, 784),
        stem_type="3x3",
        stem_chs=80,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=1024,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
    "nfnet.8":
    # ~4.2M parameters
    nfnet.NfCfg(
        depths=(1, 4, 8, 2),
        channels=(144, 288, 576, 864),
        stem_type="3x3",
        stem_chs=88,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=1152,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
    "nfnet.9":
    # ~5.3M parameters
    nfnet.NfCfg(
        depths=(1, 4, 9, 2),
        channels=(160, 320, 640, 960),
        stem_type="3x3",
        stem_chs=96,
        group_size=8,
        bottle_ratio=0.25,
        extra_conv=True,
        num_features=1280,
        act_layer="silu",
        attn_layer="eca",
        attn_kwargs={},
    ),
}
