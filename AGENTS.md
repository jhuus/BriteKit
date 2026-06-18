# AGENTS.md

This file provides guidance to coding agents when working in this repository.

## Project Overview

BriteKit (Bioacoustic Recognizer Toolkit) is a Python package for building and deploying deep learning bioacoustic recognizers. It supports the full ML pipeline: downloading recordings from Xeno-Canto, iNaturalist, and YouTube; managing training data in SQLite; training/testing/tuning/calibrating models; and running inference.

The package is installed as the `britekit` CLI entrypoint (`britekit --help`). Every CLI command also has a matching Python API function in `britekit.commands`.

## Development Environment

This project uses [Hatch](https://hatch.pypa.io/) with the `britekit` environment:

```bash
hatch run britekit:test              # run all tests
hatch run britekit:test tests/test_audio.py  # run a single test file
hatch run britekit:format            # black formatting
hatch run britekit:lint              # ruff linting
hatch run britekit:typecheck         # mypy type checking
hatch run britekit:check             # format check + lint + typecheck together
```

Tests require `src` on the Python path (configured in `pytest.ini`).

Before finishing code changes, prefer running the narrowest relevant tests first,
then `hatch run britekit:check` when the change touches package code. The mypy
configuration ignores missing imports, so a clean typecheck does not mean all
third-party interactions are fully typed.

Do not regenerate or manually edit generated reference docs unless the task is
documentation-related. `scripts/generate_readme.py` is used for README/API/CLI
reference generation.

## Architecture

### Package layout (`src/britekit/`)

- **`cli.py`** — Click group that registers all CLI commands. Each command is defined in `commands/_<name>.py` and exposed both as a Click command (`_<name>_cmd`) and as a plain Python function imported in `commands/__init__.py`.
- **`core/`** — Domain logic:
  - `audio.py` / `audio_util.py` — Audio loading (via torchaudio) and mel/log/linear spectrogram generation. The `Audio` class has an optional spectrogram cache (`use_spec_cache`) that concatenates slices into one large spectrogram for speed.
  - `base_config.py` — All configuration as OmegaConf structured dataclasses (`AudioConfig`, `TrainingConfig`, `MiscConfig`, etc.). Config is a process-level singleton accessed via `config_loader.get_config()`.
  - `config_loader.py` — Singleton `BaseConfig` instance; merge YAML overrides with `get_config(cfg_path)`.
  - `predictor.py` — `Predictor` wraps one model or an ensemble directory for inference; returns `Label` objects with score/start/end.
  - `analyzer.py` — `Analyzer` adds multi-threading and multi-recording orchestration on top of `Predictor`.
  - `trainer.py` / `tuner.py` — PyTorch Lightning–based training and hyperparameter tuning.
  - `data_module.py` / `dataset.py` — Lightning `DataModule` and dataset implementations.
  - `augmentation.py` — Audio/spectrogram augmentation pipeline.
  - `pickler.py` / `reextractor.py` — Serialize training data to/from pickle files.
- **`models/`** — All model definitions are `pl.LightningModule` subclasses extending `BaseModel` (`base_model.py`). Backbone variants: `effnet.py`, `bknet.py`, `timm_model.py`, `dla.py`, `hgnet.py`, `gernet.py`, `vovnet.py`. Classifier heads are created by `head_factory.py`. `model_loader.py` handles `.ckpt` and `.onnx` loading. `model_inspector.py` reads metadata embedded in checkpoints.
- **`training_db/`** — SQLite-backed training database (`TrainingDatabase`). Stores classes, recordings, segments, and their labels. `extractor.py` extracts spectrogram segments; `training_data_provider.py` provides batches for training.
- **`occurrence_db/`** — Separate SQLite database for occurrence/detection results used during inference and testing.
- **`testing/`** — Tester classes (`per_recording_tester.py`, `per_segment_tester.py`, `iou_tester.py`, etc.) for evaluating model performance.
- **`commands/`** — One file per CLI command. Each file exports a Click command object and a plain Python function with the same logic.

### Config system

Configuration lives in `core/base_config.py` as nested dataclasses. A YAML file (e.g. `yaml/myconfig.yaml`) can override defaults and is loaded via `get_config(cfg_path)`. The config singleton is process-global; tests should call `config_loader.set_base_config(...)` to reset it.

### Data flow for training

1. Download audio → `xeno`, `inat`, `youtube` commands
2. Add to training DB → `add-class`, `add-src`, `extract-all` etc.
3. Serialize to pickle → `pickle-train`
4. Train → `train` (uses `Trainer` → `DataModule` → `BaseModel`)
5. Test/tune/calibrate → `rpt-test`, `tune`, `calibrate`
6. Export → `ckpt-onnx`

### Data flow for inference

`analyze` command → `Analyzer` → `Predictor` → `Audio` (spectrogram) → model forward pass → `Label` objects → output CSV/SQLite.
