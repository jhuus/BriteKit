#!/usr/bin/env python3

"""Public BriteKit API.

Exports are loaded on first access so importing an inference submodule does not
also import the CLI, testing, and training stacks.
"""

from importlib import import_module
from typing import Any

from .__about__ import __version__

# SPDX-FileCopyrightText: 2025-present Jan Huus <jhuus1@gmail.com>
# SPDX-License-Identifier: MIT

_EXPORTS = {
    "commands": ("britekit.commands", None),
    "get_config": ("britekit.core.config_loader", "get_config"),
    "load_new_model": ("britekit.models.model_loader", "load_new_model"),
    "load_from_checkpoint": ("britekit.models.model_loader", "load_from_checkpoint"),
    "util": ("britekit.core.util", None),
    "Analyzer": ("britekit.core.analyzer", "Analyzer"),
    "Audio": ("britekit.core.audio", "Audio"),
    "BaseConfig": ("britekit.core.base_config", "BaseConfig"),
    "Extractor": ("britekit.training_db.extractor", "Extractor"),
    "OccurrenceDatabase": (
        "britekit.occurrence_db.occurrence_db",
        "OccurrenceDatabase",
    ),
    "OccurrenceDatabaseV2": (
        "britekit.occurrence_db.occurrence_database_v2",
        "OccurrenceDatabaseV2",
    ),
    "OccurrenceDataProvider": (
        "britekit.occurrence_db.occurrence_data_provider",
        "OccurrenceDataProvider",
    ),
    "OccurrencePickleProvider": (
        "britekit.occurrence_db.occurrence_pickle",
        "OccurrencePickleProvider",
    ),
    "OccurrencePickler": ("britekit.core.pickler", "OccurrencePickler"),
    "PerBlockTester": ("britekit.testing.per_block_tester", "PerBlockTester"),
    "PerRecordingTester": (
        "britekit.testing.per_recording_tester",
        "PerRecordingTester",
    ),
    "PerSegmentTester": ("britekit.testing.per_segment_tester", "PerSegmentTester"),
    "Predictor": ("britekit.core.predictor", "Predictor"),
    "Trainer": ("britekit.core.trainer", "Trainer"),
    "TrainingDatabase": ("britekit.training_db.training_db", "TrainingDatabase"),
    "TrainingDataProvider": (
        "britekit.training_db.training_data_provider",
        "TrainingDataProvider",
    ),
    "Tuner": ("britekit.core.tuner", "Tuner"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load and cache a public export when it is first requested."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
