"""Save reproducible samples of frame labels and compare later rounds."""

from collections import defaultdict
from copy import deepcopy
import html
import json
import logging
from pathlib import Path
import pickle
import random
from typing import Optional

import click

from britekit.core import util
from britekit.core.base_config import AudioConfig, BaseConfig
from britekit.core.config_loader import get_config


def _select_sample(data, labels, per_class, seed):
    """Sample recordings first, then a segment, without looking at scores."""
    required = (
        "class_codes",
        "class_names",
        "spec_values",
        "spec_class_indexes",
        "spec_segment_ids",
        "spec_recording_ids",
    )
    if any(key not in data for key in required):
        raise ValueError(f"Training pickle must contain {required}")
    count = len(data["spec_values"])
    if any(len(data[key]) != count for key in required[3:]):
        raise ValueError("Training pickle has inconsistent sample array lengths")
    if len(set(data["spec_segment_ids"])) != count:
        raise ValueError("Training pickle contains duplicate segment IDs")
    codes = data["class_codes"]
    if len(codes) != len(data["class_names"]) or len(set(codes)) != len(codes):
        raise ValueError("Training pickle has inconsistent class metadata")
    groups = defaultdict(lambda: defaultdict(list))
    for i, segment_id in enumerate(data["spec_segment_ids"]):
        if segment_id not in labels:
            continue
        indexes = data["spec_class_indexes"][i]
        if len(indexes) != 1 or not 0 <= indexes[0] < len(codes):
            raise ValueError(f"Segment {segment_id} must have one valid class label")
        groups[codes[indexes[0]]][int(data["spec_recording_ids"][i])].append(i)
    rng = random.Random(seed)
    selected = []
    for code in sorted(groups):
        recordings = groups[code]
        recording_ids = sorted(recordings)
        rng.shuffle(recording_ids)
        chosen = [
            rng.choice(sorted(recordings[r], key=lambda i: data["spec_segment_ids"][i]))
            for r in recording_ids[:per_class]
        ]
        # Small classes can still supply several examples from one recording.
        if len(chosen) < per_class:
            chosen_set = set(chosen)
            remaining = sorted(
                (
                    i
                    for group in recordings.values()
                    for i in group
                    if i not in chosen_set
                ),
                key=lambda i: data["spec_segment_ids"][i],
            )
            chosen += rng.sample(
                remaining, min(per_class - len(chosen), len(remaining))
            )
        selected.extend(sorted(chosen, key=lambda i: data["spec_segment_ids"][i]))
    if not selected:
        raise ValueError("No labeled segments match the training pickle")
    return {
        key: (
            [data[key][i] for i in selected] if key.startswith("spec_") else data[key]
        )
        for key in required
    }


def _selected_labels(labels, segment_ids, num_frames):
    import numpy as np

    selected = {}
    for segment_id in segment_ids:
        if segment_id not in labels:
            raise ValueError(f"Frame labels are missing selected segment {segment_id}")
        curve = np.asarray(labels[segment_id], dtype=np.float32)
        if curve.shape != (num_frames,):
            raise ValueError(
                f"Segment {segment_id} has frame shape {curve.shape}; expected ({num_frames},). "
                "Check spec_duration and sed_fps."
            )
        if not np.isfinite(curve).all() or ((curve < 0) | (curve > 1)).any():
            raise ValueError(f"Segment {segment_id} has invalid frame probabilities")
        selected[int(segment_id)] = curve.copy()
    return selected


def _plot_frame_spec(spec, curve, previous, record, cfg, output):
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 5.4), dpi=120, layout="constrained")
    FigureCanvasAgg(fig)
    spec_ax, label_ax = fig.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    duration = cfg.audio.spec_duration
    height = spec.shape[0]
    spec_ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
        extent=(0, duration, 0, height),
        interpolation="nearest",
    )
    audio = cfg.audio
    if audio.freq_scale == "mel":
        import librosa

        frequencies = librosa.mel_frequencies(
            n_mels=height, fmin=audio.min_freq, fmax=audio.max_freq
        )
    elif audio.freq_scale == "log":
        frequencies = np.geomspace(audio.min_freq, audio.max_freq, height)
    else:
        frequencies = np.linspace(audio.min_freq, audio.max_freq, height)
    locations = np.linspace(0, height - 1, 5).astype(int)
    spec_ax.set_yticks(
        locations + 0.5, [f"{frequencies[i] / 1000:.1f}" for i in locations]
    )
    spec_ax.set_ylabel("Frequency (kHz)")
    spec_ax.set_title(
        f"{record['class_code']} · {record['class_name']}\n"
        f"Segment {record['segment_id']} · Recording {record['recording_id']}",
        loc="left",
        fontsize=12,
    )
    edges = np.linspace(0, duration, len(curve) + 1)
    if previous is not None:
        label_ax.stairs(
            previous,
            edges,
            baseline=None,
            color="#2864ac",
            linewidth=2,
            label="Saved baseline",
        )
    label_ax.stairs(
        curve, edges, baseline=None, color="#d46215", linewidth=2, label="Current"
    )
    label_ax.set(
        xlim=(0, duration),
        ylim=(-0.02, 1.02),
        xlabel="Time (seconds)",
        ylabel="Probability",
    )
    label_ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    label_ax.grid(alpha=0.2)
    label_ax.legend(loc="upper right", fontsize=9)
    label_ax.set_title(
        f"{len(curve)} frames · {cfg.train.sed_fps:g} frames/second",
        loc="left",
        fontsize=9,
    )
    fig.savefig(output)
    fig.clear()


def _write_gallery(destination, manifest):
    records = manifest["samples"]
    groups = defaultdict(list)
    for record in records:
        groups[record["class_code"]].append(record)
    sections = []
    links = []
    for i, (code, examples) in enumerate(groups.items()):
        links.append(f'<a href="#class-{i}">{html.escape(code)}</a>')
        cards = []
        for record in examples:
            delta = record.get("mean_absolute_change")
            comparison = (
                "" if delta is None else f" · Mean absolute change: {delta:.3f}"
            )
            path = html.escape(record["image"])
            cards.append(
                f'<figure><a href="{path}"><img loading="lazy" src="{path}" '
                f'alt="{html.escape(code)} segment {record["segment_id"]}"></a>'
                f'<figcaption>Segment {record["segment_id"]} · '
                f'Recording {record["recording_id"]}{comparison}</figcaption></figure>'
            )
        sections.append(
            f'<section id="class-{i}"><h2>{html.escape(examples[0]["class_name"])}</h2>{"".join(cards)}</section>'
        )
    title = (
        "Frame-label comparison" if manifest["baseline"] else "Frame-label inspection"
    )
    text = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
body{{font:16px system-ui,sans-serif;color:#243244;background:#f3f5f7;max-width:1100px;margin:36px auto;padding:0 24px}}
h1{{margin-bottom:8px}}a{{color:#245f9d}}nav{{display:flex;gap:12px;flex-wrap:wrap;padding:18px 0}}
figure{{margin:20px 0;background:white;padding:14px;border-radius:8px}}img{{width:100%;height:auto}}
figcaption{{padding:8px;color:#526072}}section{{scroll-margin-top:20px}}h2{{margin-top:40px}}
</style><h1>{title}</h1>
<p>{len(records)} fixed examples · {len(groups)} classes · {manifest['num_frames']} frames per segment</p>
<p>Samples were selected independently of confidence scores. Compare the probability curves with the visible vocalizations;
a confidence increase alone does not establish better localization.</p>
<p><a href="selection.json">Selection and settings</a> · <a href="labels.pkl">Saved labels</a> · <a href="sample.pkl">Saved spectrogram sample</a></p>
<nav>{' '.join(links)}</nav>{''.join(sections)}</html>
"""
    (destination / "index.html").write_text(text, encoding="utf-8")


def plot_frame(
    frame_pickle_path: str,
    output_path: str,
    train_pickle_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
    per_class: int = 5,
    seed: int = 42,
) -> None:
    """Save a fixed sample of spectrograms and soft frame labels for inspection.

    On the first run, supply --train-pickle and --cfg describing its audio settings
    and sed_fps. Sample up to per_class examples per class without using confidence
    scores, preferring different recordings. Save a self-contained review directory:
    selection.json (IDs/settings), sample.pkl (selected training spectrograms),
    labels.pkl (selected frame labels in pickle-frame format), and an HTML gallery.

    For the next round, supply --baseline with that directory and the new frame
    pickle. Reuse the exact sample and saved display settings, overlaying old and
    new labels. No training pickle or config is needed. Missing IDs or different
    frame lengths are errors; examples are never silently replaced or resampled.
    Output directories must be new to preserve earlier inspection results.

    Args:
    - frame_pickle_path (str): Frame-label pickle from pickle-frame-infer or pickle-frame.
    - output_path (str): New directory for the review bundle and plots.
    - train_pickle_path (str, optional): Training spectrogram pickle for the initial sample.
    - cfg_path (str, optional): YAML for the initial spectrogram display and frame grid.
    - baseline_path (str, optional): Previous plot-frame output directory to compare with.
    - per_class (int): Initial sample size per class (default 5).
    - seed (int): Initial sampling seed (default 42).
    """
    import numpy as np

    if per_class < 1:
        raise ValueError("per_class must be positive")
    if bool(train_pickle_path) == bool(baseline_path):
        raise ValueError("Supply either train_pickle_path or baseline_path")
    if baseline_path and cfg_path:
        raise ValueError("A baseline supplies the saved config; omit cfg_path")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    with open(frame_pickle_path, "rb") as file:
        labels = pickle.load(file)
    if not isinstance(labels, dict):
        raise ValueError("Frame-label pickle must be a dictionary keyed by segment ID")
    previous = None
    if baseline_path:
        baseline = Path(baseline_path)
        manifest = json.loads((baseline / "selection.json").read_text())
        if manifest["format_version"] != 1:
            raise ValueError("Unsupported frame inspection format version")
        cfg = BaseConfig()
        cfg.audio = AudioConfig(**manifest["audio_config"])
        cfg.train.sed_fps = manifest["sed_fps"]
        with (baseline / "sample.pkl").open("rb") as file:
            sample = pickle.load(file)
        if sample["spec_segment_ids"] != [r["segment_id"] for r in manifest["samples"]]:
            raise ValueError("Baseline sample and selection segment IDs differ")
        with (baseline / "labels.pkl").open("rb") as file:
            previous = pickle.load(file)
    else:
        cfg = deepcopy(get_config(cfg_path))
        assert train_pickle_path is not None
        logging.info("Loading training spectrograms from %s", train_pickle_path)
        with open(train_pickle_path, "rb") as file:
            data = pickle.load(file)
        sample = _select_sample(data, labels, per_class, seed)
        del data
        manifest = dict(
            format_version=1,
            train_pickle=str(Path(train_pickle_path).resolve()),
            seed=seed,
            per_class=per_class,
            sampling="random recordings, then random segments",
            audio_config=util.cfg_to_pure(cfg.audio),
            sed_fps=cfg.train.sed_fps,
        )
    num_frames = round(cfg.audio.spec_duration * cfg.train.sed_fps)
    if num_frames < 1:
        raise ValueError("spec_duration and sed_fps must define at least one frame")
    selected = _selected_labels(labels, sample["spec_segment_ids"], num_frames)
    if previous is not None:
        previous = _selected_labels(previous, sample["spec_segment_ids"], num_frames)
    if cfg.audio.convert_to_db and cfg.audio.decibels:
        raise ValueError("convert_to_db requires linear stored spectrograms")
    specs = []
    for compressed in sample["spec_values"]:
        spec = util.expand_spectrogram(compressed, cfg=cfg).squeeze(0)
        if cfg.audio.convert_to_db:
            from britekit.core.audio_util import convert_to_db

            spec = convert_to_db(
                spec, cfg.audio.power, cfg.audio.top_db, db_power=cfg.audio.db_power
            )
        specs.append(spec)
    manifest.update(
        frame_pickle=str(Path(frame_pickle_path).resolve()),
        num_frames=num_frames,
        baseline=None if baseline_path is None else str(Path(baseline_path).resolve()),
        samples=[],
    )
    destination.mkdir(parents=True)
    (destination / "plots").mkdir()
    logging.info("Plotting %d selected segments", len(selected))
    for i, (segment_id, spec) in enumerate(zip(sample["spec_segment_ids"], specs)):
        class_index = sample["spec_class_indexes"][i][0]
        record = dict(
            segment_id=int(segment_id),
            recording_id=int(sample["spec_recording_ids"][i]),
            class_code=sample["class_codes"][class_index],
            class_name=sample["class_names"][class_index],
            image=f"plots/segment-{segment_id}.png",
        )
        curve = selected[segment_id]
        old_curve = None if previous is None else previous[segment_id]
        if old_curve is not None:
            record["mean_absolute_change"] = float(np.mean(np.abs(curve - old_curve)))
        _plot_frame_spec(
            spec, curve, old_curve, record, cfg, destination / record["image"]
        )
        manifest["samples"].append(record)
    for name, value in [("sample.pkl", sample), ("labels.pkl", selected)]:
        with (destination / name).open("wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    (destination / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_gallery(destination, manifest)
    logging.info("Saved %d examples to %s", len(selected), destination / "index.html")


@click.command(
    name="plot-frame",
    short_help="Inspect a fixed sample of frame labels and compare rounds.",
    help=util.cli_help_from_doc(plot_frame.__doc__),
)
@click.argument("frame_pickle_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--train-pickle",
    "train_pickle_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Training pickle for the first round.",
)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True, file_okay=False),
    help="Previous plot-frame output directory; reuse its sample and settings.",
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Initial spectrogram and frame-grid settings.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(file_okay=False),
    help="New output directory.",
)
@click.option("--per-class", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def _plot_frame_cmd(**kwargs):
    util.set_logging()
    plot_frame(**kwargs)
