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
            )

        self.backbone.head = cast(hgnet.ClassifierHead, nn.Identity())


# Model size is most affected by number of classes for smaller models.
# For the smallest HgNet models, the default head is disproportionately large.
# Other head types such as "basic" generate a much smaller head.
MODEL_REGISTRY = {
    "hgnet.1":
    # ~460K parameters with 50 classes (~90K with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [16, 24],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [24, 24, 48, 1, False, True, 3, 2],
        "stage2": [48, 32, 96, 1, True, True, 3, 2],
        "stage3": [96, 48, 96, 1, True, True, 5, 2],
        "stage4": [96, 64, 128, 1, True, True, 5, 2],
    },
    "hgnet.2":
    # ~740K parameters with 50 classes (~230K with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, True, 3, 2],
        "stage2": [64, 48, 128, 1, True, True, 3, 2],
        "stage3": [128, 80, 160, 1, True, True, 5, 3],
        "stage4": [160, 96, 192, 1, True, True, 5, 3],
    },
    "hgnet.3":
    # ~1.3M parameters with 50 classes (~510K with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [24, 40],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [40, 48, 96, 1, False, True, 3, 2],
        "stage2": [96, 64, 160, 1, True, True, 3, 3],
        "stage3": [160, 96, 256, 1, True, True, 5, 3],
        "stage4": [256, 160, 320, 1, True, True, 5, 3],
    },
    "hgnet.4":
    # ~1.8M parameters (~800K with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 48],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [48, 64, 128, 1, False, True, 3, 2],
        "stage2": [128, 80, 192, 1, True, True, 3, 3],
        "stage3": [192, 128, 320, 1, True, True, 5, 3],
        "stage4": [320, 192, 416, 1, True, True, 5, 3],
    },
    "hgnet.5":
    # ~2.7M parameters (~1.4M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 64],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [64, 72, 144, 1, False, True, 3, 2],
        "stage2": [144, 112, 256, 1, True, True, 3, 3],
        "stage3": [256, 176, 416, 1, True, True, 5, 3],
        "stage4": [416, 256, 576, 1, True, True, 5, 3],
    },
    "hgnet.6":
    # ~3.4M parameters (~1.8M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 72, 160, 1, False, True, 3, 2],
        "stage2": [160, 112, 304, 1, True, True, 3, 3],
        "stage3": [304, 176, 480, 1, True, True, 5, 3],
        "stage4": [480, 256, 768, 1, True, True, 5, 3],
    },
    "hgnet.7":
    # this is hgnetv2_b0, with ~4.0M parameters (~1.9M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [16, 16],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 128, 1024, 1, True, True, 5, 3],
    },
    "hgnet.8":
    # this is hgnetv2_b1, with ~4.3M parameters (~2.3M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 3],
        "stage2": [64, 48, 256, 1, True, False, 3, 3],
        "stage3": [256, 96, 512, 2, True, True, 5, 3],
        "stage4": [512, 192, 1024, 1, True, True, 5, 3],
    },
    "hgnet.9":
    # ~5.4M parameters (~3.0M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 40],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [40, 40, 80, 1, False, False, 3, 3],
        "stage2": [80, 64, 288, 1, True, False, 3, 3],
        "stage3": [288, 112, 576, 2, True, True, 5, 4],
        "stage4": [576, 224, 1152, 1, True, True, 5, 3],
    },
    "hgnet.10":
    # ~6.7M parameters (~4.1M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 48],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [48, 64, 128, 1, False, False, 5, 3],
        "stage2": [128, 96, 320, 1, True, False, 3, 3],
        "stage3": [320, 144, 640, 2, True, True, 5, 4],
        "stage4": [640, 256, 1280, 1, True, True, 5, 3],
    },
    "hgnet.11":
    # ~7.2M parameters (~4.6M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 56],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [56, 72, 144, 1, False, True, 5, 4],
        "stage2": [144, 112, 256, 1, True, True, 3, 4],
        "stage3": [256, 176, 704, 1, True, True, 3, 5],
        "stage4": [704, 416, 1216, 1, True, True, 3, 5],
    },
    "hgnet.12":
    # ~7.9M parameters (~5.2M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 60],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [60, 80, 160, 1, False, True, 5, 4],
        "stage2": [160, 128, 272, 1, True, True, 3, 4],
        "stage3": [272, 192, 736, 1, True, True, 3, 5],
        "stage4": [736, 448, 1280, 1, True, True, 3, 5],
    },
    "hgnet.13":
    # ~8.3M parameters (~5.5M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 64],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [64, 80, 160, 1, False, True, 5, 4],
        "stage2": [160, 128, 288, 1, True, True, 3, 4],
        "stage3": [288, 192, 768, 1, True, True, 3, 5],
        "stage4": [768, 448, 1344, 1, True, True, 3, 5],
    },
    "hgnet.14":
    # ~9.0M parameters (~6.1M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 56],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [56, 72, 152, 1, False, True, 5, 4],
        "stage2": [152, 112, 288, 1, True, True, 3, 5],
        "stage3": [288, 192, 800, 1, True, True, 5, 5],
        "stage4": [800, 480, 1408, 1, True, True, 5, 5],
    },
    "hgnet.15":
    # ~9.6M parameters (~6.6M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 48],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [48, 64, 144, 1, False, True, 5, 4],
        "stage2": [144, 96, 288, 1, True, True, 3, 5],
        "stage3": [288, 192, 832, 1, True, True, 5, 5],
        "stage4": [832, 512, 1472, 1, True, True, 5, 5],
    },
    "hgnet.16":
    # ~10.8M parameters (~7.5M with basic head)
    {
        "stem_type": "v2",
        "stem_chs": [32, 56],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [56, 72, 160, 1, False, True, 5, 4],
        "stage2": [160, 112, 320, 1, True, True, 3, 5],
        "stage3": [320, 208, 896, 1, True, True, 5, 5],
        "stage4": [896, 544, 1568, 1, True, True, 5, 5],
    },
}
