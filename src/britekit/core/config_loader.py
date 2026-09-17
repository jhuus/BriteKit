#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
from typing import cast, Optional
from britekit.core.base_config import BaseConfig

_base_config: Optional[BaseConfig] = None


def get_config(cfg_path: Optional[str] = None) -> BaseConfig:
    from omegaconf import OmegaConf, DictConfig

    if cfg_path is None:
        return get_config_with_dict()
    else:
        yaml_cfg = cast(DictConfig, OmegaConf.load(cfg_path))
        return get_config_with_dict(yaml_cfg)


def get_config_with_dict(cfg_dict=None) -> BaseConfig:
    from omegaconf import OmegaConf

    global _base_config
    if _base_config is None:
        _base_config = OmegaConf.structured(BaseConfig())

    # allow late merges/overrides even if already initialized
    if cfg_dict is not None:
        _base_config = cast(
            BaseConfig, OmegaConf.merge(_base_config, OmegaConf.create(cfg_dict))
        )
    if _base_config.audio.spec_bits not in (8, 16):
        raise ValueError("audio.spec_bits must be 8 or 16")
    if _base_config.audio.convert_to_db:
        from britekit.core.audio_util import validate_db_power

        validate_db_power(_base_config.audio.db_power)
    return _base_config


def set_base_config(cfg: BaseConfig) -> None:
    global _base_config
    _base_config = cfg
