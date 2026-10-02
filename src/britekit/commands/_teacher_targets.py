#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer heavyweight imports to improve --help performance.
import hashlib
import logging
from copy import deepcopy
from numbers import Integral
from pathlib import Path
import pickle
import time
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def teacher_targets(
    train_pickle_path: str,
    checkpoint_path: str,
    output_path: str,
    cfg_path: Optional[str] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> None:
    """
    Generate soft segment and frame targets from a checkpoint or ensemble.

    The input must be a BriteKit training pickle containing stable segment IDs.
    Stored spectrograms must match the teachers' source representation. Linear
    features are converted to dB per teacher when its checkpoint requests it,
    using its power, top_db, and db_power settings, without training augmentation.
    For an ensemble directory, probabilities are averaged across all checkpoints.
    SED frame probabilities are stored when the teachers provide them. Calibration
    and application-level filtering are not applied.

    Args:
    - train_pickle_path (str): Path to the BriteKit training pickle.
    - checkpoint_path (str): Path to a teacher checkpoint or ensemble directory.
    - output_path (str): Path for the generated teacher-target pickle.
    - cfg_path (str, optional): Path to YAML configuration overrides.
    - batch_size (int): Number of spectrograms per inference batch.
    - device (str, optional): Inference device, such as cpu, cuda, or mps.
    """
    _generate_targets(
        train_pickle_path, checkpoint_path, output_path, cfg_path, batch_size, device
    )


def _generate_targets(
    train_pickle_path: str,
    checkpoint_path: str,
    output_path: str,
    cfg_path: Optional[str],
    batch_size: int,
    device: Optional[str],
    *,
    frame_labels_only: bool = False,
) -> None:
    """Shared inference for distillation targets and legacy frame-label pickles."""
    import numpy as np

    from britekit.core.audio_util import convert_to_db
    from britekit.models import model_loader

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    source_path = Path(train_pickle_path)
    teacher_path = Path(checkpoint_path)
    destination = Path(output_path)

    logging.info("Loading training spectrograms from %s", source_path)
    with source_path.open("rb") as file:
        training_data = pickle.load(file)

    required_keys: tuple[str, ...] = ("class_codes", "spec_values", "spec_segment_ids")
    if frame_labels_only:
        required_keys += ("spec_class_indexes",)
    missing = [key for key in required_keys if key not in training_data]
    if missing:
        raise ValueError(f"Training pickle is missing required keys: {missing}")

    class_codes = list(training_data["class_codes"])
    specs = training_data["spec_values"]
    segment_ids = list(training_data["spec_segment_ids"])
    if len(specs) != len(segment_ids):
        raise ValueError(
            "Training pickle has different numbers of spectrograms and segment IDs"
        )
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("Training pickle contains duplicate segment IDs")

    label_indexes = []
    if frame_labels_only:
        if not specs:
            raise ValueError("Training pickle contains no spectrograms")
        if not class_codes or len(class_codes) != len(set(class_codes)):
            raise ValueError("Training pickle must have unique, nonempty class codes")
        labels = training_data["spec_class_indexes"]
        if len(labels) != len(specs):
            raise ValueError(
                "Training pickle has different numbers of labels and spectrograms"
            )
        for segment_id, indexes in zip(segment_ids, labels):
            if not isinstance(segment_id, Integral) or isinstance(segment_id, bool):
                raise ValueError("Frame-label pickle requires integer segment IDs")
            if len(indexes) != 1:
                raise ValueError(
                    f"Segment {segment_id} must have exactly one class label; "
                    "the pickle-frame format stores one curve per segment"
                )
            index = indexes[0]
            if (
                not isinstance(index, Integral)
                or isinstance(index, bool)
                or not 0 <= int(index) < len(class_codes)
            ):
                raise ValueError(
                    f"Invalid class index for segment {segment_id}: {index}"
                )
            label_indexes.append(int(index))

    cfg = deepcopy(get_config(cfg_path))
    # Distillation targets are the teachers' uncalibrated sigmoid probabilities.
    cfg.infer.scaling_coefficient = 1.0
    cfg.infer.scaling_intercept = 0.0
    inference_device = device or util.get_device()
    checkpoint_files = (
        [teacher_path]
        if teacher_path.is_file()
        else sorted(teacher_path.glob("*.ckpt"))
    )
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {teacher_path}")
    if destination.resolve() in {
        source_path.resolve(),
        *(p.resolve() for p in checkpoint_files),
    }:
        raise ValueError(
            "Output path must differ from the training pickle and checkpoints"
        )

    logging.info(
        "Generating targets for %d spectrograms and %d classes using %d teacher checkpoint(s) on %s",
        len(specs),
        len(class_codes),
        len(checkpoint_files),
        inference_device,
    )
    models = []
    model_class_indexes = []
    for model_index, checkpoint_file in enumerate(checkpoint_files, start=1):
        logging.info(
            "Loading teacher %d/%d: %s",
            model_index,
            len(checkpoint_files),
            checkpoint_file.name,
        )
        model = model_loader.load_from_checkpoint(
            str(checkpoint_file), apply_training_config=False
        ).eval()
        # Keep each teacher's preprocessing independent of other checkpoints
        # and of the caller's global config.
        model.apply_training_config(deepcopy(cfg))
        model = model.to(inference_device)
        model_codes = list(model.train_class_codes)
        if frame_labels_only:
            missing_codes = sorted(set(class_codes) - set(model_codes))
            if len(model_codes) != len(set(model_codes)) or missing_codes:
                raise ValueError(
                    f"Checkpoint {checkpoint_file.name} must have unique class codes "
                    f"covering the training classes; missing: {missing_codes}"
                )
            model_class_indexes.append(
                np.asarray([model_codes.index(class_codes[i]) for i in label_indexes])
            )
        elif model_codes != class_codes:
            raise ValueError(
                "Teacher ensemble checkpoints do not have identical class codes and ordering"
            )
        models.append(model)

    audio = models[0].cfg.audio
    height = audio.spec_height
    width = audio.spec_width
    source_keys = ["spec_height", "spec_width", "power", "decibels"]
    if frame_labels_only:
        source_keys += [
            "spec_duration",
            "sampling_rate",
            "win_length",
            "n_fft",
            "min_freq",
            "max_freq",
            "freq_scale",
            "mel_norm",
            "log_freq_gain",
        ]
        if audio.decibels:
            source_keys += ["top_db", "db_power"]
    for model in models:
        settings = model.cfg.audio
        for key in source_keys:
            if getattr(settings, key) != getattr(audio, key):
                raise ValueError(f"Teacher ensemble source features differ: {key}")
        if settings.convert_to_db and settings.decibels:
            raise ValueError(
                "convert_to_db requires a linear pickle and audio.decibels=false"
            )
        if frame_labels_only and model.cfg.train.sed_fps != models[0].cfg.train.sed_fps:
            raise ValueError(
                "Frame-label ensemble checkpoints must have matching sed_fps"
            )

    probabilities = (
        None
        if frame_labels_only
        else np.empty((len(specs), len(class_codes)), dtype=np.float32)
    )
    frame_probabilities = None
    frame_labels = {}
    num_batches = (len(specs) + batch_size - 1) // batch_size
    progress_interval = max(1, num_batches // 100)
    inference_start = time.monotonic()
    for start in range(0, len(specs), batch_size):
        end = min(start + batch_size, len(specs))
        batch = np.empty((end - start, 1, height, width), dtype=np.float32)
        for index, compressed_spec in enumerate(specs[start:end]):
            expanded = util.expand_spectrogram(compressed_spec, cfg=models[0].cfg)
            batch[index] = expanded.reshape(1, height, width)
        predictions = []
        for model_index, model in enumerate(models):
            settings = model.cfg.audio
            model_batch = batch
            if settings.convert_to_db:
                model_batch = convert_to_db(
                    batch,
                    settings.power,
                    settings.top_db,
                    db_power=settings.db_power,
                )
            segment_scores, frames = model.predict(model_batch, inference_device)
            if frame_labels_only:
                if frames is None:
                    raise ValueError(
                        "Frame-label generation requires SED frame outputs from every checkpoint"
                    )
                frames = np.asarray(frames)
                expected = (
                    end - start,
                    len(model.train_class_codes),
                    round(settings.spec_duration * model.cfg.train.sed_fps),
                )
                if frames.shape != expected:
                    raise ValueError(
                        f"Unexpected frame output shape: {frames.shape}; expected {expected}"
                    )
                # Select the known class, retaining only one curve per source
                # segment instead of allocating all classes for the full dataset.
                frames = frames[
                    np.arange(end - start), model_class_indexes[model_index][start:end]
                ][:, None, :]
                if (
                    not np.isfinite(frames).all()
                    or (frames < 0).any()
                    or (frames > 1).any()
                ):
                    raise ValueError(
                        "Frame outputs must be finite probabilities in [0, 1]"
                    )
            predictions.append((segment_scores, frames))
        if probabilities is not None:
            scores = [prediction[0] for prediction in predictions]
            probabilities[start:end] = np.mean(scores, axis=0, dtype=np.float32)

        frame_scores = [prediction[1] for prediction in predictions]
        has_frame_scores = [frame_output is not None for frame_output in frame_scores]
        if any(has_frame_scores) and not all(has_frame_scores):
            raise ValueError(
                "Teacher ensemble mixes models with and without frame outputs"
            )
        if all(has_frame_scores):
            valid_frame_scores = [
                frame_output
                for frame_output in frame_scores
                if frame_output is not None
            ]
            first_shape = valid_frame_scores[0].shape
            if any(
                frame_output.shape != first_shape
                for frame_output in valid_frame_scores[1:]
            ):
                raise ValueError(
                    "Teacher ensemble checkpoints have different frame output shapes"
                )
            if first_shape[:2] != (
                end - start,
                1 if frame_labels_only else len(class_codes),
            ):
                raise ValueError(
                    f"Unexpected teacher frame output shape: {first_shape}"
                )
            averaged_frames = np.asarray(
                np.mean(valid_frame_scores, axis=0, dtype=np.float32), dtype=np.float32
            )
            if frame_labels_only:
                for segment_id, curve in zip(
                    segment_ids[start:end], averaged_frames[:, 0]
                ):
                    frame_labels[int(segment_id)] = curve.copy()
            elif frame_probabilities is None:
                frame_probabilities = np.empty(
                    (len(specs), len(class_codes), first_shape[2]), dtype=np.float16
                )
                logging.info(
                    "Teacher frame targets have %d frames per spectrogram",
                    first_shape[2],
                )
            elif frame_probabilities.shape[2] != first_shape[2]:
                raise ValueError("Teacher frame output length changed between batches")
            if frame_probabilities is not None:
                frame_probabilities[start:end] = averaged_frames
        elif frame_probabilities is not None:
            raise ValueError("Teacher frame outputs disappeared between batches")
        batch_number = start // batch_size + 1
        if (
            batch_number == 1
            or batch_number % progress_interval == 0
            or end == len(specs)
        ):
            elapsed = time.monotonic() - inference_start
            rate = end / elapsed if elapsed > 0 else 0.0
            logging.info(
                "Teacher inference: %d/%d spectrograms (%.1f%%), %.1f spectrograms/s",
                end,
                len(specs),
                100 * end / len(specs) if specs else 100.0,
                rate,
            )

    if frame_labels_only:
        _write_targets(destination, frame_labels)
        logging.info(
            "Wrote soft frame labels for %d segments (%d frames, %s fps, %s seconds) to %s",
            len(frame_labels),
            len(next(iter(frame_labels.values()))),
            models[0].cfg.train.sed_fps,
            audio.spec_duration,
            destination,
        )
        return

    logging.info("Computing source and checkpoint fingerprints")
    output = {
        "format_version": 2,
        "class_codes": class_codes,
        "segment_ids": segment_ids,
        "probabilities": probabilities,
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "teacher": {
            "path": str(teacher_path),
            "ensemble_method": "mean_probability",
            "checkpoints": [
                {
                    "name": path.name,
                    "sha256": _sha256(path),
                }
                for path in checkpoint_files
            ],
        },
    }
    if frame_probabilities is not None:
        output["frame_probabilities"] = frame_probabilities

    _write_targets(destination, output)
    assert probabilities is not None
    logging.info(
        "Wrote teacher targets with segment shape %s and frame shape %s to %s",
        probabilities.shape,
        None if frame_probabilities is None else frame_probabilities.shape,
        destination,
    )


def _write_targets(destination: Path, output) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as file:
            pickle.dump(output, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


@click.command(
    name="teacher-targets",
    short_help="Generate soft training targets from a model ensemble.",
    help=util.cli_help_from_doc(teacher_targets.__doc__),
)
@click.argument("train_pickle_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--checkpoints",
    "checkpoint_path",
    type=click.Path(exists=True),
    required=True,
    help="Teacher checkpoint or directory containing an ensemble.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Output teacher-target pickle.",
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to YAML configuration overrides.",
)
@click.option(
    "--batch-size", type=click.IntRange(min=1), default=256, show_default=True
)
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]))
def _teacher_targets_cmd(
    train_pickle_path: str,
    checkpoint_path: str,
    output_path: str,
    cfg_path: Optional[str],
    batch_size: int,
    device: Optional[str],
) -> None:
    util.set_logging()
    teacher_targets(
        train_pickle_path,
        checkpoint_path,
        output_path,
        cfg_path,
        batch_size,
        device,
    )
