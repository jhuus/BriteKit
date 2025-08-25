from typing import cast, Optional

from omegaconf import OmegaConf, DictConfig
from britekit.core.base_config import BaseConfig, FunctionConfig

# private singleton cache
_base_config: Optional[BaseConfig] = None
_func_config: Optional[FunctionConfig] = None


def get_config(cfg_path: Optional[str] = None) -> tuple[BaseConfig, FunctionConfig]:
    if cfg_path is None:
        return get_config_with_dict()
    else:
        yaml_cfg = cast(DictConfig, OmegaConf.load(cfg_path))
        return get_config_with_dict(yaml_cfg)


def get_config_with_dict(
    cfg_dict: Optional[DictConfig] = None,
) -> tuple[BaseConfig, FunctionConfig]:
    global _base_config, _func_config
    if _base_config is None:
        _base_config = OmegaConf.structured(BaseConfig())
        _func_config = FunctionConfig()
        if cfg_dict is not None:
            _base_config = cast(
                BaseConfig, OmegaConf.merge(_base_config, OmegaConf.create(cfg_dict))
            )

    return cast(tuple[BaseConfig, FunctionConfig], (_base_config, _func_config))
