#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
from typing import Any, List, Optional

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
    from britekit.models.mobilenet import MobileNet
    from britekit.models.nfnet import NfNetModel
    from britekit.models.repvit import RepVitModel
    from britekit.models.resnet import ResNetModel
    from britekit.models.vovnet import VovNetModel

    cfg = get_config()
    device = get_device()

    # create a dict of optional keyword arguments
    kwargs: dict[str, Any] = {"temporal_pooling": cfg.train.temporal_pooling}
    if cfg.train.head_type == "prototype_sed":
        kwargs["prototypes_per_class"] = cfg.train.prototypes_per_class
    if cfg.train.head_type in ("temporal_sed", "prototype_sed"):
        kwargs["lse_temp"] = cfg.train.lse_temp
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
    elif model_type.startswith("mobilenet"):
        model_class = MobileNet
    elif model_type.startswith("nfnet"):
        model_class = NfNetModel
    elif model_type.startswith("repvit"):
        model_class = RepVitModel
    elif model_type.startswith("resnet"):
        model_class = ResNetModel
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


def load_from_checkpoint(
    checkpoint_path: str,
    multi_label: Optional[bool] = None,
    apply_training_config: bool = True,
):
    # defer these imports to improve --help performance
    import torch

    from britekit.models.timm_model import TimmModel
    from britekit.models.bknet import BKNetModel
    from britekit.models.dla import DlaModel
    from britekit.models.effnet import EffNetModel
    from britekit.models.gernet import GerNetModel
    from britekit.models.hgnet import HGNetModel
    from britekit.models.mobilenet import MobileNet
    from britekit.models.nfnet import NfNetModel
    from britekit.models.repvit import RepVitModel
    from britekit.models.resnet import ResNetModel
    from britekit.models.vovnet import VovNetModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = get_device()
    model_class: Any = None
    strict = ckpt["hyper_parameters"].get("head_type") == "prototype_sed"
    if "model_type" in ckpt["hyper_parameters"]:
        model_type = ckpt["hyper_parameters"]["model_type"]
        if model_type.startswith("timm."):
            if multi_label is None:
                model = TimmModel.load_from_checkpoint(checkpoint_path, strict=strict)
            else:
                model = TimmModel.load_from_checkpoint(
                    checkpoint_path, multi_label=multi_label, strict=strict
                )
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
        elif model_type.startswith("mobilenet"):
            model_class = MobileNet
        elif model_type.startswith("nfnet"):
            model_class = NfNetModel
        elif model_type.startswith("repvit"):
            model_class = RepVitModel
        elif model_type.startswith("resnet"):
            model_class = ResNetModel
        elif model_type.startswith("vovnet"):
            model_class = VovNetModel
        else:
            raise ModelError(f'Unable to load model with unknown type "{model_type}"')

        if not model_type.startswith("timm."):
            if multi_label is None:
                model = model_class.load_from_checkpoint(checkpoint_path, strict=strict)
            else:
                model = model_class.load_from_checkpoint(
                    checkpoint_path, multi_label=multi_label, strict=strict
                )

        if apply_training_config:
            model.apply_training_config(get_config())
        return model.to(device)
    else:
        raise ModelError("Checkpoint file has no model_type information.")


def initialize_backbone(model, checkpoint_path: str) -> None:
    """Copy compatible backbone weights only; keep the new head and run config.

    Species ordering is checked even though the head is not copied, making
    accidental changes to the baseline experiment explicit. Built-in classifier
    weights in a timm source backbone may be discarded.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    for key in ("model_type", "train_class_names", "train_class_codes"):
        if hp.get(key) != getattr(model, key):
            raise ModelError(f"Backbone initialization requires matching {key}")
    # Reusing weights with a different frontend would confound this experiment.
    source_audio = ckpt.get("training_cfg", {}).get("audio", {})
    from britekit.core.base_config import AudioConfig
    from dataclasses import asdict

    for key, default in asdict(AudioConfig()).items():
        if key in (
            "use_spec_cache",
            "chunks_per_spec",
            "choose_channel",
            "check_seconds",
            "spec_bits",
        ):
            continue
        source_value = source_audio.get(key, default)
        if source_value != getattr(model.cfg.audio, key):
            raise ModelError(f"Backbone initialization requires matching audio.{key}")
    if model.backbone is None:
        raise ModelError("Target model has no backbone")
    source = {
        k[len("backbone.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("backbone.")
    }
    target = model.backbone.state_dict()
    missing = set(target) - set(source)
    extra = set(source) - set(target)
    allowed_extra = ("head.", "classifier.", "fc.", "global_pool.")
    unexpected = [k for k in extra if not k.startswith(allowed_extra)]
    mismatched = [
        k for k in target if k in source and source[k].shape != target[k].shape
    ]
    if missing or unexpected or mismatched:
        raise ModelError(
            f"Incompatible backbone: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shape_mismatch={mismatched}"
        )
    model.backbone.load_state_dict({k: source[k] for k in target}, strict=True)
