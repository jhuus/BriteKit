# Defer some imports to improve initialization performance.
from typing import cast, Optional
from britekit.core.base_config import BaseConfig, FunctionConfig

_base_config: Optional[BaseConfig] = None
_func_config: Optional[FunctionConfig] = None


def get_config(cfg_path: Optional[str] = None) -> BaseConfig:
    from omegaconf import OmegaConf, DictConfig

    if cfg_path is None:
        return get_config_with_dict()
    else:
        yaml_cfg = cast(DictConfig, OmegaConf.load(cfg_path))
        return get_config_with_dict(yaml_cfg)


def get_func_config() -> FunctionConfig:
    global _func_config
    if _func_config is None:
        _func_config = FunctionConfig()

    return _func_config


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
    return _base_config


def set_base_config(cfg: BaseConfig) -> None:
    global _base_config
    _base_config = cfg


def set_func_config(cfg: FunctionConfig) -> None:
    global _func_config
    _func_config = cfg
