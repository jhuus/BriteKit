#!/usr/bin/env python3

from typing import List, Optional

from timm.models import resnet
from torch import nn

from britekit.models.base_model import BaseModel
from britekit.models.head_factory import make_head


class _TemporalBasicBlock(resnet.BasicBlock):
    """Optionally shorten conv2 along time, retaining timm's block behavior."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *args,
        short_temporal_channels: tuple[int, ...] = (),
        **kwargs,
    ):
        super().__init__(inplanes, planes, *args, **kwargs)
        if planes in short_temporal_channels:
            conv = self.conv2
            # Spectrogram axes are frequency, time. Keep frequency context and
            # spatial alignment; only remove conv2's temporal neighbors.
            self.conv2 = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=(3, 1),
                stride=(conv.stride[0], conv.stride[1]),
                padding=(conv.dilation[0], 0),
                dilation=(conv.dilation[0], 1),
                bias=False,
                device=conv.weight.device,
                dtype=conv.weight.dtype,
            )
            # ResNet initializes this convolution, along with all other layers,
            # after constructing its blocks. Keep its Kaiming/zero-last-BN scheme.


class ResNetModel(BaseModel):
    """ResNet-10T variants separating channel width from temporal context.

    All variants use one basic block per stage and the same deep tiered stem,
    stride-32 output, and 512 output channels. With a 192 x 384 spectrogram,
    their feature maps are 512 x 6 x 12.
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
        prototypes_per_class: int = 5,
        lse_temp: float = 0.5,
        temporal_pooling: str = "logsumexp",
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
            prototypes_per_class,
            lse_temp,
            temporal_pooling,
        )

        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        channels, short_temporal = MODEL_REGISTRY[model_type]
        two_way = kwargs.pop("two_way", True)
        block = _TemporalBasicBlock if short_temporal else resnet.BasicBlock
        self.backbone = resnet.ResNet(
            # timm annotates this as a block instance, but requires a class.
            block=block,  # type: ignore[arg-type]
            layers=(1, 1, 1, 1),
            stem_width=32,
            stem_type="deep_tiered",
            avg_down=True,
            channels=channels,
            # These widths occur only in stages 3 and 4 in every preset.
            block_args=(
                {"short_temporal_channels": channels[2:]} if short_temporal else None
            ),
            in_chans=1,
            num_classes=self.num_classes if head_type is None else 0,
            global_pool="avg" if head_type is None else "",
            **kwargs,
        )
        if head_type is None:
            self.head = nn.Identity()
        else:
            self.head = make_head(
                head_type,
                self.backbone.num_features,
                hidden_channels,
                self.num_classes,
                drop_rate=kwargs.get("drop_rate", 0.0),
                lse_temp=lse_temp,
                temporal_pooling=temporal_pooling,
                prototypes_per_class=prototypes_per_class,
                two_way=two_way,
            )


# ((stage widths), shorten conv2 in stages 3 and 4). Parameter counts exclude
# the classifier/head and use a one-channel input. All presets train from
# scratch, like BriteKit's other custom model families.
MODEL_REGISTRY = {
    "resnet.1": ((64, 128, 256, 512), False),  # 4,922,056; timm resnet10t
    "resnet.2": ((96, 192, 384, 512), False),  # 7,083,144; wider
    "resnet.3": ((64, 128, 256, 512), True),  # 2,955,976; shorter time context
    "resnet.4": ((96, 192, 384, 512), True),  # 4,625,544; wider + shorter
    "resnet.5": ((112, 224, 448, 512), True),  # 5,606,440; wider + shorter
    "resnet.6": ((128, 256, 512, 512), True),  # 6,686,152; wider + shorter
}
