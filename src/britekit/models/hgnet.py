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
# Setting non_sed_head_type = "basic" generates a much smaller head.
MODEL_REGISTRY = {
    "hgnet.1":
    # ~360K parameters with 50 classes
    {
        "stem_type": "v2",
        "stem_chs": [16, 16],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 32, 1, False, False, 3, 1],
        "stage2": [32, 32, 64, 1, True, False, 3, 1],
        "stage3": [64, 64, 64, 2, True, False, 3, 1],
        "stage4": [64, 64, 64, 1, True, True, 5, 1],
    },
    "hgnet.2":
    # ~580K parameters with 50 classes
    {
        "stem_type": "v2",
        "stem_chs": [16, 16],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 32, 1, False, False, 3, 1],
        "stage2": [32, 32, 64, 1, True, False, 3, 1],
        "stage3": [64, 64, 128, 2, True, False, 3, 1],
        "stage4": [128, 64, 128, 1, True, True, 5, 1],
    },
    "hgnet.3":
    # ~1.0M parameters with 50 classes
    {
        "stem_type": "v2",
        "stem_chs": [16, 16],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 32, 1, False, False, 3, 1],
        "stage2": [32, 32, 64, 1, True, False, 3, 1],
        "stage3": [64, 128, 128, 2, True, False, 3, 1],
        "stage4": [128, 64, 256, 1, True, True, 5, 1],
    },
    "hgnet.4":
    # ~1.6M parameters
    {
        "stem_type": "v2",
        "stem_chs": [16, 16],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 1],
        "stage2": [64, 32, 128, 1, True, False, 3, 1],
        "stage3": [128, 64, 256, 2, True, True, 5, 1],
        "stage4": [256, 64, 512, 1, True, True, 5, 1],
    },
    "hgnet.5":
    # ~2.8M parameters
    {
        "stem_type": "v2",
        "stem_chs": [16, 16],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 128, 1, True, False, 3, 3],
        "stage3": [128, 64, 384, 2, True, True, 5, 3],
        "stage4": [384, 96, 768, 1, True, True, 5, 3],
    },
    "hgnet.6":
    # ~3.1M parameters
    {
        "stem_type": "v2",
        "stem_chs": [16, 16],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 96, 768, 1, True, True, 5, 3],
    },
    "hgnet.7":
    # this is hgnetv2_b0, with ~4.0M parameters
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
    # this is hgnetv2_b1, with ~4.3M parameters
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
    # ~4.6M parameters
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 4],
        "stage2": [64, 48, 256, 1, True, False, 3, 4],
        "stage3": [256, 96, 512, 2, True, True, 5, 4],
        "stage4": [512, 192, 1024, 1, True, True, 5, 4],
    },
    "hgnet.10":
    # ~5.7M parameters
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 4],
        "stage2": [64, 48, 256, 1, True, False, 3, 4],
        "stage3": [256, 96, 512, 3, True, True, 5, 4],
        "stage4": [512, 192, 1024, 1, True, True, 5, 4],
    },
    "hgnet.11":
    # ~6.1M parameters
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 96, 1, False, False, 3, 4],
        "stage2": [96, 64, 384, 1, True, False, 3, 4],
        "stage3": [384, 128, 512, 3, True, True, 5, 4],
        "stage4": [512, 192, 1024, 1, True, True, 5, 4],
    },
    "hgnet.12":
    # ~6.7M parameters
    {
        "stem_type": "v2",
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 3],
        "stage2": [64, 48, 256, 1, True, False, 3, 3],
        "stage3": [256, 128, 768, 1, True, True, 5, 3],
        "stage4": [768, 256, 1536, 1, True, True, 5, 3],
    },
}
