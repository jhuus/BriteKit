import copy
from typing import List, Optional

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

        self.backbone.head = nn.Identity()


# Model size is most affected by number of classes for smaller models
MODEL_REGISTRY = {
    "vovnet.1":
    # ~400K parameters with 50 classes
    dict(
        stem_chs=[32, 32, 32],
        stage_conv_chs=[64, 64, 96, 128],
        stage_out_chs=[64, 96, 128, 160],
        layer_per_block=1,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.2":
    # ~650K parameters with 50 classes
    dict(
        stem_chs=[32, 32, 32],
        stage_conv_chs=[64, 96, 128, 160],
        stage_out_chs=[96, 128, 160, 192],
        layer_per_block=1,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.3":
    # ~1.1M parameters with 50 classes
    dict(
        stem_chs=[32, 32, 32],
        stage_conv_chs=[96, 128, 160, 192],
        stage_out_chs=[96, 128, 256, 384],
        layer_per_block=1,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.4":
    # ~2.1M parameters
    dict(
        stem_chs=[32, 32, 64],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 256, 384, 512],
        layer_per_block=1,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.5":
    # ~3.0M parameters
    dict(
        stem_chs=[32, 32, 32],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 256, 384, 384],
        layer_per_block=1,
        block_per_stage=[1, 1, 1, 2],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.6":
    # ~3.7M parameters
    dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 256, 384, 512],
        layer_per_block=2,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.7":
    # ~4.4M parameters
    dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 256, 384, 512],
        layer_per_block=2,
        block_per_stage=[1, 2, 1, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.8":
    # ~5.5M parameters
    dict(
        stem_chs=[32, 32, 64],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 256, 384, 384],
        layer_per_block=2,
        block_per_stage=[1, 2, 2, 1],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.9":
    # ~6.2M parameters
    dict(
        stem_chs=[32, 32, 64],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 256, 384, 384],
        layer_per_block=2,
        block_per_stage=[1, 1, 2, 2],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.10":
    # ~7.6M parameters (they get much slower with layers_per_block=3 though)
    dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 256, 384, 512],
        layer_per_block=3,
        block_per_stage=[1, 1, 1, 2],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
    "vovnet.11":
    # ~9.3M parameters
    dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[128, 256, 384, 512],
        layer_per_block=3,
        block_per_stage=[1, 1, 2, 2],
        residual=True,
        depthwise=False,
        attn="eca",
    ),
}
