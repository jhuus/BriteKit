import copy
from typing import cast, List, Optional

from timm.layers import NormMlpClassifierHead
from timm.models import byobnet
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class GerNetModel(BaseModel):
    """
    Scaled version of timm gernet, where model_size parameter defines the scaling.
    Paper: `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
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
        self.backbone = byobnet.ByobNet(
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

        self.backbone.head = cast(NormMlpClassifierHead, nn.Identity())


# Model size is most affected by number of classes for smaller models
MODEL_REGISTRY = {
    "gernet.1":
    # ~430K parameters with 50 classes
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=3, c=64, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=64, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
    "gernet.2":
    # ~640K parameters with 50 classes
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=3, c=64, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=2, c=128, s=2, gs=1, br=3.0),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=64, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
    "gernet.3":
    # ~892K parameters with 50 classes
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=3, c=128, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=2, c=128, s=2, gs=1, br=3.0),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=128, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
    "gernet.4":
    # ~1.5M parameters
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=3, c=128, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=2, c=256, s=2, gs=1, br=3.0),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=128, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
    "gernet.5":
    # ~2.5M parameters
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=3, c=192, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=2, c=344, s=2, gs=1, br=3.0),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=192, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
    "gernet.6":
    # ~3.3M parameters
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=3, c=256, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=2, c=384, s=2, gs=1, br=3.0),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=256, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
    "gernet.7":
    # ~4.5M parameters
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=7, c=320, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=2, c=440, s=2, gs=1, br=3.0),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=256, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
    "gernet.8":
    # ~5.5M parameters
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=7, c=348, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=2, c=512, s=2, gs=1, br=3.0),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=256, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
    "gernet.9":
    # ~6.4M parameters
    byobnet.ByoModelCfg(
        blocks=(
            byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
            byobnet.ByoBlockCfg(type="bottle", d=7, c=384, s=2, gs=0, br=1 / 4),
            byobnet.ByoBlockCfg(type="bottle", d=2, c=560, s=2, gs=1, br=3.0),
            byobnet.ByoBlockCfg(type="bottle", d=1, c=256, s=1, gs=1, br=3.0),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),
}
