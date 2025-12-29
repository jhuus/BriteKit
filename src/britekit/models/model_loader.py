#!/usr/bin/env python3

from typing import Any, List, Optional

import torch

from britekit.core.config_loader import get_config
from britekit.core.exceptions import InputError, ModelError
from britekit.core.util import get_device


def load_new_model(
    train_class_names: List[str],
    train_class_codes: List[str],
    train_class_alt_names: List[str],
    train_class_alt_codes: List[str],
    num_train_specs: int,
):
    # defer these imports to improve --help performance
    from britekit.models.timm_model import TimmModel
    from britekit.models.bknet import BKNetModel
    from britekit.models.dla import DlaModel
    from britekit.models.effnet import EffNetModel
    from britekit.models.gernet import GerNetModel
    from britekit.models.hgnet import HGNetModel
    from britekit.models.vovnet import VovNetModel

    cfg = get_config()
    device = get_device()

    # create a dict of optional keyword arguments
    kwargs = {}
    if cfg.train.drop_rate is not None:
        kwargs.update(dict(drop_rate=cfg.train.drop_rate))

    if cfg.train.drop_path_rate is not None:
        kwargs.update(dict(drop_path_rate=cfg.train.drop_path_rate))

    # create model corresponding to specified type
    model_class: Any = None
    model_type = cfg.train.model_type
    if model_type.startswith("timm."):
        return TimmModel(
            model_type,
            cfg.train.head_type,
            cfg.train.hidden_channels,
            train_class_names,
            train_class_codes,
            train_class_alt_names,
            train_class_alt_codes,
            num_train_specs,
            cfg.train.multi_label,
            **kwargs,
        ).to(device)
    elif model_type.startswith("bk"):
        model_class = BKNetModel
    elif model_type.startswith("dla"):
        model_class = DlaModel
    elif model_type.startswith("effnet"):
        model_class = EffNetModel
    elif model_type.startswith("gernet"):
        model_class = GerNetModel
    elif model_type.startswith("hgnet"):
        model_class = HGNetModel
    elif model_type.startswith("vovnet"):
        model_class = VovNetModel
    else:
        raise InputError(f"Invalid model type = {model_type}")

    return model_class(
        model_type,
        cfg.train.head_type,
        cfg.train.hidden_channels,
        train_class_names,
        train_class_codes,
        train_class_alt_names,
        train_class_alt_codes,
        num_train_specs,
        cfg.train.multi_label,
        **kwargs,
    ).to(device)


def load_from_checkpoint(checkpoint_path: str, multi_label: Optional[bool] = None):
    # defer these imports to improve --help performance
    from britekit.models.timm_model import TimmModel
    from britekit.models.bknet import BKNetModel
    from britekit.models.dla import DlaModel
    from britekit.models.effnet import EffNetModel
    from britekit.models.gernet import GerNetModel
    from britekit.models.hgnet import HGNetModel
    from britekit.models.vovnet import VovNetModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = get_device()
    model_class: Any = None
    if "model_type" in ckpt["hyper_parameters"]:
        model_type = ckpt["hyper_parameters"]["model_type"]
        if model_type.startswith("timm."):
            if multi_label is None:
                return TimmModel.load_from_checkpoint(checkpoint_path, strict=False).to(
                    device
                )
            else:
                return TimmModel.load_from_checkpoint(
                    checkpoint_path, multi_label=multi_label, strict=False
                ).to(device)
        elif model_type.startswith("bk"):
            model_class = BKNetModel
        elif model_type.startswith("dla"):
            model_class = DlaModel
        elif model_type.startswith("effnet"):
            model_class = EffNetModel
        elif model_type.startswith("gernet"):
            model_class = GerNetModel
        elif model_type.startswith("hgnet"):
            model_class = HGNetModel
        elif model_type.startswith("vovnet"):
            model_class = VovNetModel
        else:
            raise ModelError(f'Unable to load model with unknown type "{model_type}"')

        if multi_label is None:
            return model_class.load_from_checkpoint(checkpoint_path, strict=False).to(
                device
            )
        else:
            return model_class.load_from_checkpoint(
                checkpoint_path, multi_label=multi_label, strict=False
            ).to(device)
    else:
        raise ModelError("Checkpoint file has no model_type information.")
