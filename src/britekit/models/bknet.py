#!/usr/bin/env python3

"""
BKNet model for BriteKit - wraps BKNetBaseModel with classifier heads.
See comments at top of bknet_base.py.
"""

import copy
from typing import cast, List, Optional, Type

import torch.nn as nn

from britekit.models.base_model import BaseModel
from britekit.models.bknet_base import BKNetBaseModel
from britekit.models.head_factory import make_head


class BKNetModel(BaseModel):
    """
    BKNet (BriteKit Network) model.

    A configurable GENet-style CNN with:
    - BasicBlocks for early stages
    - Bottleneck blocks for later stages
    - Standard convolutions (not depthwise separable)
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

        # Extract parameters from config
        act_name = config.get("act_layer", "relu")
        act_layer = _get_act_layer(cast(str, act_name))
        final_ch = config.get("final_ch", 1920)
        use_se = config.get("use_se", False)
        se_ratio = config.get("se_ratio", 0.25)

        # Create backbone
        self.backbone = BKNetBaseModel(
            cfg=config,
            num_classes=self.num_classes,
            in_chans=1,
            act_layer=act_layer,
            final_ch=cast(int, final_ch),
            use_se=use_se,
            se_ratio=se_ratio,
            **kwargs,
        )

        # Select head
        if head_type is None:
            # use the default BKNet head
            self.head = nn.Sequential(
                copy.deepcopy(self.backbone.global_pool),
                copy.deepcopy(self.backbone.drop),
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

        self.backbone.global_pool = cast(nn.Sequential, nn.Identity())
        self.backbone.drop = cast(nn.Dropout, nn.Identity())
        self.backbone.fc = cast(nn.Linear, nn.Identity())
        self.backbone.flatten = nn.Identity()


def _get_act_layer(name: str) -> Type[nn.Module]:
    """Get activation layer class by name."""
    activations = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Choose from {list(activations.keys())}"
        )
    return activations[name.lower()]


# =============================================================================
# Model Registry
# =============================================================================
# Configuration keys:
#   stem_ch: int           - output channels of stem conv
#   stage_out_chs: list    - output channels per stage
#   stage_mid_chs: list    - middle channels for bottleneck (0 = BasicBlock)
#   num_blocks: list       - number of blocks per stage
#   stage_strides: list    - stride for first block of each stage (default: [2,2,...,2,1])
#   final_ch: int          - final 1x1 conv expansion (optional)
#   act_layer: str         - activation function name (default: "relu")
#
# GENet architecture (from gernet-backbone-shapes.txt analysis):
#   Input: (1, 1, 192, 384)
#   Stem: 1 -> 13 channels, stride 2 -> (1, 13, 96, 192)
#   Stage 0: BasicBlock, 13 -> 48, stride 2, 1 block -> (1, 48, 48, 96)
#   Stage 1: BasicBlock, 48 -> 48, stride 2, 3 blocks -> (1, 48, 24, 48)
#   Stage 2: Bottleneck, 48 -> 256, mid=64, stride 2, 3 blocks -> (1, 256, 12, 24)
#   Stage 3: Bottleneck, 256 -> 384, mid=1152, stride 2, 2 blocks -> (1, 384, 6, 12)
#   Stage 4: Bottleneck, 384 -> 256, mid=768, stride 1, 1 block -> (1, 256, 6, 12)
#   Final: 1x1 conv 256 -> 1920 -> (1, 1920, 6, 12)
# =============================================================================

MODEL_REGISTRY = {
    # ~650K parameters + head size
    "bknet.1": dict(
        stem_ch=13,
        stage_out_chs=[32, 32, 112, 176, 112],
        stage_mid_chs=[0, 0, 32, 704, 352],
        num_blocks=[1, 2, 2, 1, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=1024,
        act_layer="relu",
    ),
    # ~1.7M parameters + head size
    "bknet.2": dict(
        stem_ch=16,  # slightly wider stem, still cheap
        stage_out_chs=[40, 40, 176, 256, 160],
        stage_mid_chs=[0, 0, 48, 960, 480],
        num_blocks=[1, 3, 3, 2, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=1536,
        act_layer="relu",
    ),
    # ~3.0M parameters + head size
    "bknet.3": dict(
        stem_ch=13,
        stage_out_chs=[48, 48, 256, 384, 224],
        stage_mid_chs=[0, 0, 64, 1152, 576],
        num_blocks=[1, 3, 3, 2, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=1920,
        act_layer="relu",
    ),
    # ~3.7M + head size
    "bknet.4": dict(
        stem_ch=20,
        stage_out_chs=[56, 56, 288, 432, 256],
        stage_mid_chs=[0, 0, 72, 1296, 640],
        num_blocks=[1, 3, 3, 2, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=1920,
        act_layer="relu",
    ),
    # ~4.4M + head size
    "bknet.5": dict(
        stem_ch=24,
        stage_out_chs=[64, 64, 320, 448, 272],
        stage_mid_chs=[0, 0, 96, 1408, 704],
        num_blocks=[1, 3, 3, 2, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=1920,
        act_layer="relu",
    ),
    # ~5.6M + head size
    "bknet.6": dict(
        stem_ch=24,
        stage_out_chs=[64, 64, 320, 452, 272],
        stage_mid_chs=[0, 0, 96, 1356, 678],
        num_blocks=[1, 3, 3, 3, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=1920,
        act_layer="relu",
    ),
    # ~5.8M + head size
    "bknet.7": dict(
        stem_ch=24,
        stage_out_chs=[64, 64, 320, 464, 280],
        stage_mid_chs=[0, 0, 96, 1392, 696],
        num_blocks=[1, 3, 3, 3, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=1920,
        act_layer="relu",
    ),
    # ~6.1M parameters + head size
    "bknet.8": dict(
        stem_ch=24,
        stage_out_chs=[64, 64, 320, 480, 288],
        stage_mid_chs=[0, 0, 96, 1440, 720],
        num_blocks=[1, 3, 3, 3, 1],  # keep stage 4 at 1 block like others
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=1920,
        act_layer="relu",
    ),
    # ~6.6M + head size
    "bknet.9": dict(
        stem_ch=24,
        stage_out_chs=[64, 64, 352, 496, 304],
        stage_mid_chs=[0, 0, 104, 1488, 744],
        num_blocks=[1, 3, 3, 3, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=2048,
        act_layer="relu",
    ),
    # ~7.5M + head size
    "bknet.10": dict(
        stem_ch=28,
        stage_out_chs=[72, 72, 384, 528, 320],
        stage_mid_chs=[0, 0, 112, 1584, 792],
        num_blocks=[1, 3, 3, 3, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=2048,
        act_layer="relu",
    ),
    # ~8.4M + head size
    "bknet.11": dict(
        stem_ch=28,
        stage_out_chs=[72, 72, 416, 560, 344],
        stage_mid_chs=[0, 0, 120, 1680, 840],
        num_blocks=[1, 3, 3, 3, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=2048,
        act_layer="relu",
    ),
    # ~9.5M + head size
    "bknet.12": dict(
        stem_ch=32,
        stage_out_chs=[80, 80, 448, 592, 368],
        stage_mid_chs=[0, 0, 128, 1776, 888],
        num_blocks=[1, 3, 3, 3, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=2176,
        act_layer="relu",
    ),
    # ~10.6M + head size
    "bknet.13": dict(
        stem_ch=32,
        stage_out_chs=[80, 80, 480, 624, 392],
        stage_mid_chs=[0, 0, 136, 1872, 936],
        num_blocks=[1, 3, 3, 3, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=2304,
        act_layer="relu",
    ),
    # ~11.8M + head size
    "bknet.14": dict(
        stem_ch=32,
        stage_out_chs=[88, 88, 512, 656, 416],
        stage_mid_chs=[0, 0, 144, 1968, 984],
        num_blocks=[1, 3, 3, 3, 1],
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=2432,
        act_layer="relu",
    ),
    # ~13.4M + head size
    "bknet.15": dict(
        stem_ch=36,
        stage_out_chs=[96, 96, 544, 688, 440],
        stage_mid_chs=[0, 0, 152, 2064, 1032],
        num_blocks=[1, 3, 4, 3, 1],  # extra block in stage 2
        stage_strides=[2, 2, 2, 2, 1],
        stage_depthwise=[False, False, False, True, True],
        final_ch=2560,
        act_layer="relu",
    ),
}
