#!/usr/bin/env python3

import copy
from typing import cast, List, Optional

from timm.models import hgnet
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class HGNetModel(BaseModel):
    """Scaled version of timm hgnet_v2, where model_size parameter defines the scaling."""

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
        self.backbone = hgnet.HighPerfGpuNet(
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

        self.backbone.head = cast(hgnet.ClassifierHead, nn.Identity())


# Sizes below include the backbone only and exclude the classifier head.
MODEL_REGISTRY = {
    "hgnet.1":
    # Backbone is 0.09M parameters
    {
        "stem_type": "v2",
        "stem_chs": [16, 24],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [24, 24, 48, 1, False, True, 3, 2],
        "stage2": [48, 32, 96, 1, True, True, 3, 2],
        "stage3": [96, 48, 96, 1, True, True, 5, 2],
        "stage4": [96, 64, 128, 1, True, True, 5, 2],
    },
    "hgnet.2":
    # Backbone is 0.23M parameters
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, True, 3, 2],
        "stage2": [64, 48, 128, 1, True, True, 3, 2],
        "stage3": [128, 80, 160, 1, True, True, 5, 3],
        "stage4": [160, 96, 192, 1, True, True, 5, 3],
    },
    "hgnet.3":
    # Backbone is 0.50M parameters
    {
        "stem_type": "v2",
        "stem_chs": [24, 40],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [40, 48, 96, 1, False, True, 3, 2],
        "stage2": [96, 64, 160, 1, True, True, 3, 3],
        "stage3": [160, 96, 256, 1, True, True, 5, 3],
        "stage4": [256, 160, 320, 1, True, True, 5, 3],
    },
    "hgnet.4":
    # Backbone is 0.78M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 48],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [48, 64, 128, 1, False, True, 3, 2],
        "stage2": [128, 80, 192, 1, True, True, 3, 3],
        "stage3": [192, 128, 320, 1, True, True, 5, 3],
        "stage4": [320, 192, 416, 1, True, True, 5, 3],
    },
    "hgnet.5":
    # Backbone is 1.35M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 64],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [64, 72, 144, 1, False, True, 3, 2],
        "stage2": [144, 112, 256, 1, True, True, 3, 3],
        "stage3": [256, 176, 416, 1, True, True, 5, 3],
        "stage4": [416, 256, 576, 1, True, True, 5, 3],
    },
    "hgnet.6":
    # Backbone is 1.73M parameters
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 72, 160, 1, False, True, 3, 2],
        "stage2": [160, 112, 304, 1, True, True, 3, 3],
        "stage3": [304, 176, 480, 1, True, True, 5, 3],
        "stage4": [480, 256, 768, 1, True, True, 5, 3],
    },
    "hgnet.7":
    # Backbone is 1.85M parameters (timm hgnetv2_b0)
    {
        "stem_type": "v2",
        "stem_chs": [16, 16],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 128, 1024, 1, True, True, 5, 3],
    },
    "hgnet.8":
    # Backbone is 2.20M parameters (timm hgnetv2_b1)
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 3],
        "stage2": [64, 48, 256, 1, True, False, 3, 3],
        "stage3": [256, 96, 512, 2, True, True, 5, 3],
        "stage4": [512, 192, 1024, 1, True, True, 5, 3],
    },
    "hgnet.9":
    # Backbone is 2.96M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 40],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [40, 40, 80, 1, False, False, 3, 3],
        "stage2": [80, 64, 288, 1, True, False, 3, 3],
        "stage3": [288, 112, 576, 2, True, True, 5, 4],
        "stage4": [576, 224, 1152, 1, True, True, 5, 3],
    },
    "hgnet.10":
    # Backbone is 3.61M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 56],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [56, 72, 136, 1, False, True, 3, 3],
        "stage2": [136, 104, 240, 1, True, True, 3, 3],
        "stage3": [240, 168, 672, 1, True, True, 5, 4],
        "stage4": [672, 392, 1152, 1, True, True, 5, 4],
    },
    "hgnet.11":
    # Backbone is 4.02M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 56],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [56, 72, 144, 1, False, True, 5, 3],
        "stage2": [144, 112, 256, 1, True, True, 5, 3],
        "stage3": [256, 176, 704, 1, True, True, 5, 4],
        "stage4": [704, 416, 1216, 1, True, True, 5, 4],
    },
    "hgnet.12":
    # Backbone is 4.99M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 56],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [56, 72, 144, 1, False, False, 3, 3],
        "stage2": [144, 112, 256, 1, True, False, 3, 3],
        "stage3": [256, 176, 704, 1, True, True, 5, 5],
        "stage4": [704, 416, 1216, 1, True, True, 5, 5],
    },
    "hgnet.13":
    # Backbone is 5.15M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 60],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [60, 80, 160, 1, False, True, 5, 4],
        "stage2": [160, 128, 272, 1, True, True, 3, 4],
        "stage3": [272, 192, 736, 1, True, True, 3, 5],
        "stage4": [736, 448, 1280, 1, True, True, 3, 5],
    },
    "hgnet.14":
    # Backbone is 5.43M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 64],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [64, 80, 160, 1, False, True, 5, 4],
        "stage2": [160, 128, 288, 1, True, True, 3, 4],
        "stage3": [288, 192, 768, 1, True, True, 3, 5],
        "stage4": [768, 448, 1344, 1, True, True, 3, 5],
    },
    "hgnet.15":
    # Backbone is 6.00M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 56],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [56, 72, 152, 1, False, True, 5, 4],
        "stage2": [152, 112, 288, 1, True, True, 3, 5],
        "stage3": [288, 192, 800, 1, True, True, 5, 5],
        "stage4": [800, 480, 1408, 1, True, True, 5, 5],
    },
    "hgnet.16":
    # Backbone is 6.51M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 48],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [48, 64, 144, 1, False, True, 5, 4],
        "stage2": [144, 96, 288, 1, True, True, 3, 5],
        "stage3": [288, 192, 832, 1, True, True, 5, 5],
        "stage4": [832, 512, 1472, 1, True, True, 5, 5],
    },
    "hgnet.17":
    # Backbone is 7.45M parameters
    {
        "stem_type": "v2",
        "stem_chs": [32, 56],
        "agg": "se",
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [56, 72, 160, 1, False, True, 5, 4],
        "stage2": [160, 112, 320, 1, True, True, 3, 5],
        "stage3": [320, 208, 896, 1, True, True, 5, 5],
        "stage4": [896, 544, 1568, 1, True, True, 5, 5],
    },
}
