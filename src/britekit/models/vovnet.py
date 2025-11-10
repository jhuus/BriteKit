import copy
from typing import Any, cast, List, Optional

from timm.models import vovnet
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class VovNetModel(BaseModel):
    """
    Scaled version of timm vovnet, where model_size parameter defines the scaling.
    Papers:
      `An Energy and GPU-Computation Efficient Backbone Network` - https://arxiv.org/abs/1904.09730
      `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
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
        self.backbone = vovnet.VovNet(
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
            )

        self.backbone.head = cast(Any, nn.Identity())


# Model size is most affected by number of classes for smaller models.
# Depthwise convs are used in 1–3 for efficiency; full conv from 4+
# Attention: ECA for 1–9 (light), SE for 10–14, GC for 15+
MODEL_REGISTRY = {
    "vovnet.1":
    # ~200K parameters with 50 classes
    dict(
        stem_chs=[32, 32, 32],
        stage_conv_chs=[64, 96, 128, 160],
        stage_out_chs=[64, 96, 128, 160],
        layer_per_block=1,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=True,
        attn="eca",
    ),
    "vovnet.2":
    # ~380K parameters with 50 classes
    dict(
        stem_chs=[32, 32, 32],
        stage_conv_chs=[96, 128, 160, 192],
        stage_out_chs=[96, 128, 160, 192],
        layer_per_block=1,
        block_per_stage=[1, 2, 1, 1],
        residual=True,
        depthwise=True,
        attn="eca",
    ),
    "vovnet.3":
    # ~800K parameters with 50 classes
    dict(
        stem_chs=[32, 32, 32],
        stage_conv_chs=[72, 104, 136, 168],
        stage_out_chs=[80, 120, 168, 208],
        layer_per_block=1,
        block_per_stage=[1, 2, 1, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.4":
    # ~1.0M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[80, 112, 144, 176],
        stage_out_chs=[96, 144, 192, 224],
        layer_per_block=1,
        block_per_stage=[1, 2, 1, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.5":
    # ~1.7M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[64, 88, 120, 152],
        stage_out_chs=[80, 120, 168, 216],
        layer_per_block=2,
        block_per_stage=[1, 2, 2, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.6":
    # ~2.2M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[72, 104, 136, 168],
        stage_out_chs=[88, 136, 192, 248],
        layer_per_block=2,
        block_per_stage=[1, 2, 2, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.7":
    # ~2.6M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[80, 112, 144, 176],
        stage_out_chs=[96, 160, 224, 288],
        layer_per_block=2,
        block_per_stage=[1, 2, 2, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.8":
    # ~3.2M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[96, 128, 160, 192],
        stage_out_chs=[112, 176, 240, 304],
        layer_per_block=2,
        block_per_stage=[1, 2, 2, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.9":
    # ~3.9M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[112, 144, 176, 208],
        stage_out_chs=[112, 192, 272, 352],
        layer_per_block=2,
        block_per_stage=[1, 2, 2, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.10":
    # ~4.7M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 208, 288, 368],
        layer_per_block=2,
        block_per_stage=[1, 2, 2, 1],
        residual=True,
        depthwise=False,
        attn="se",
    ),
    "vovnet.11":
    # ~5.3M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 208, 288, 368],
        layer_per_block=2,
        block_per_stage=[1, 3, 2, 1],
        residual=True,
        depthwise=False,
        attn="se",
    ),
    "vovnet.12":
    # ~5.7M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 208, 288, 368],
        layer_per_block=2,
        block_per_stage=[1, 2, 3, 1],
        residual=True,
        depthwise=False,
        attn="se",
    ),
    "vovnet.13":
    # ~6.3M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 208, 288, 368],
        layer_per_block=2,
        block_per_stage=[1, 3, 3, 1],
        residual=True,
        depthwise=False,
        attn="se",
    ),
    "vovnet.14":
    # ~7.2M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 208, 288, 368],
        layer_per_block=2,
        block_per_stage=[1, 2, 3, 2],
        residual=True,
        depthwise=False,
        attn="se",
    ),
    "vovnet.15":
    # ~8.6M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 48],
        stage_conv_chs=[144, 176, 208, 240],
        stage_out_chs=[144, 224, 320, 416],
        layer_per_block=2,
        block_per_stage=[1, 2, 3, 2],
        residual=True,
        depthwise=False,
        attn="gc",
    ),
}
