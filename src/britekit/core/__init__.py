#!/usr/bin/env python3

"""Core BriteKit modules, loaded on first access."""

from importlib import import_module
from types import ModuleType

__all__ = [
    "analyzer",
    "audio",
    "base_config",
    "config_loader",
    "pickler",
    "plot",
    "predictor",
    "reextractor",
    "trainer",
    "tuner",
    "util",
]


def __getattr__(name: str) -> ModuleType:
    """Load and cache a public core module when it is first requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
