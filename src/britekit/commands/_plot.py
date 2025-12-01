# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from pathlib import Path
from typing import Optional

import click

from britekit.core.config_loader import get_config, BaseConfig
from britekit.core import util


def _plot_recording(
    cfg: BaseConfig,
    audio,
    input_path: str,
    output_path: str,
    all: bool,
    overlap: float,
    ndims: bool,
):
    from britekit.core.plot import plot_spec

    logging.info(f'Processing "{input_path}"')
    signal, rate = audio.load(input_path)
    if signal is None:
        logging.error(f"Failed to read {input_path}")
        quit()

    recording_seconds = len(signal) / rate
    if all:
        # plot the whole recording in one spectrogram
        specs, _ = audio.get_spectrograms([0], spec_duration=recording_seconds)
        if specs is None:
            logging.error(f'Error: failed to extract spectrogram from "{input_path}".')
            quit()

        specs = specs.cpu().numpy()
        image_path = os.path.join(output_path, Path(input_path).stem + ".jpeg")
        plot_spec(
            specs[0], image_path, show_dims=not ndims, spec_duration=recording_seconds
        )
    else:
        # plot individual segments
        increment = max(0.5, cfg.audio.spec_duration - overlap)
        last_offset = max(0, recording_seconds - 0.5)
        offsets = util.get_range(0, last_offset, increment)
        specs, _ = audio.get_spectrograms(
            offsets, spec_duration=cfg.audio.spec_duration
        )
        if specs is None:
            logging.error(f'Error: failed to extract spectrogram from "{input_path}".')
            quit()

        specs = specs.cpu().numpy()
        for i, spec in enumerate(specs):
            image_path = os.path.join(
                output_path, f"{Path(input_path).stem}-{offsets[i]:.1f}.jpeg"
            )
            plot_spec(spec, image_path, show_dims=not ndims)


def plot_db(
    cfg_path: Optional[str] = None,
    class_name: str = "",
    db_path: Optional[str] = None,
    ndims: bool = False,
    max_count: Optional[float] = None,
    output_path: str = "",
    prefix: Optional[str] = None,
    power: Optional[float] = 1.0,
    spec_group: Optional[str] = None,
):
    """
    Plot spectrograms from a training database for a specific class.

    This command extracts spectrograms from the training database for a given class and
    saves them as JPEG images. It can filter recordings by filename prefix and limit the
    number of spectrograms plotted.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - class_name (str): Name of the class to plot spectrograms for (e.g., "Common Yellowthroat").
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - ndims (bool): If True, do not show time and frequency dimensions on the spectrogram plots.
    - max_count (int, optional): Maximum number of spectrograms to plot. If omitted, plots all available.
    - output_path (str): Directory where spectrogram images will be saved.
    - prefix (str, optional): Only include recordings that start with this filename prefix.
    - power (float, optional): Raise spectrograms to this power for visualization. Lower values show more detail.
    - spec_group (str, optional): Spectrogram group name to plot from. Defaults to "default".
    """
    from britekit.core.plot import plot_spec
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config(cfg_path)
    if power is not None:
        cfg.audio.power = power

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if db_path is None:
        db_path = cfg.train.train_db

    if spec_group is None:
        spec_group = "default"

    with TrainingDatabase(db_path) as db:
        results = db.get_spectrogram_by_class(class_name, spec_group=spec_group)
        num_plotted = 0
        prev_filename = None
        if prefix is not None:
            prefix = prefix.lower()  # do case-insensitive compares

        for r in results:
            if (
                prefix is not None
                and len(prefix) > 0
                and not r.filename.lower().startswith(prefix)
            ):
                continue

            spec_path = os.path.join(
                output_path, f"{Path(r.filename).stem}-{r.offset:.2f}.jpeg"
            )
            if not os.path.exists(spec_path):
                if r.filename != prev_filename:
                    logging.info(f"Processing {r.filename}")
                    prev_filename = r.filename

                spec = util.expand_spectrogram(r.value)
                plot_spec(spec, spec_path, show_dims=not ndims)
                num_plotted += 1

            if max_count is not None and num_plotted >= max_count:
                break

        logging.info(f"Plotted {num_plotted} spectrograms")


@click.command(
    name="plot-db",
    short_help="Plot spectrograms from a database.",
    help=util.cli_help_from_doc(plot_db.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    required=False,
    help="Path to YAML file defining config overrides.",
)
@click.option(
    "--name", "class_name", required=True, help="Plot spectrograms for this class."
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option(
    "--ndims",
    "ndims",
    is_flag=True,
    help="If specified, do not show seconds on x-axis and frequencies on y-axis.",
)
@click.option(
    "--max",
    "max_count",
    type=int,
    required=False,
    help="Max number of spectrograms to plot.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory.",
)
@click.option(
    "--prefix",
    "prefix",
    type=str,
    required=False,
    help="Only include recordings that start with this prefix.",
)
@click.option(
    "--power",
    "power",
    type=float,
    required=False,
    help="Raise spectrograms to this power. Lower values show more detail.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    help="Spectrogram group name. Defaults to 'default'.",
)
def _plot_db_cmd(
    cfg_path: str,
    class_name: str,
    db_path: Optional[str],
    ndims: bool,
    max_count: Optional[float],
    output_path: str,
    prefix: Optional[str],
    power: Optional[float],
    spec_group: Optional[str],
):
    util.set_logging()
    plot_db(
        cfg_path,
        class_name,
        db_path,
        ndims,
        max_count,
        output_path,
        prefix,
        power,
        spec_group,
    )


def plot_dir(
    cfg_path: Optional[str] = None,
    ndims: bool = False,
    input_path: str = "",
    output_path: str = "",
    all: bool = False,
    overlap: float = 0.0,
    power: float = 1.0,
):
    """
    Plot spectrograms for all audio recordings in a directory.

    This command processes all audio files in a directory and generates spectrogram images.
    It can either plot each recording as a single spectrogram or break recordings into
    overlapping segments.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - ndims (bool): If True, do not show time and frequency dimensions on the spectrogram plots.
    - input_path (str): Directory containing audio recordings to process.
    - output_path (str): Directory where spectrogram images will be saved.
    - all (bool): If True, plot each recording as one spectrogram. If False, break into segments.
    - overlap (float): Spectrogram overlap in seconds when breaking recordings into segments. Default is 0.
    - power (float): Raise spectrograms to this power for visualization. Lower values show more detail. Default is 1.0.
    """
    from britekit.core.audio import Audio

    cfg = get_config(cfg_path)
    if power is not None:
        cfg.audio.power = power

    if overlap is None:
        overlap = cfg.infer.overlap

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    audio_paths = util.get_audio_files(input_path)
    if len(audio_paths) == 0:
        logging.error(f'Error: no recordings found in "{input_path}".')
        quit()

    audio = Audio(cfg=cfg)
    for audio_path in audio_paths:
        _plot_recording(cfg, audio, audio_path, output_path, all, overlap, ndims)


@click.command(
    name="plot-dir",
    short_help="Plot spectrograms from a directory of recordings.",
    help=util.cli_help_from_doc(plot_dir.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    required=False,
    help="Path to YAML file defining config overrides.",
)
@click.option(
    "--ndims",
    "ndims",
    is_flag=True,
    help="If specified, do not show seconds on x-axis and frequencies on y-axis.",
)
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to input directory.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory.",
)
@click.option(
    "--all",
    "all",
    is_flag=True,
    help="If specified, plot whole recordings in one spectrogram each. Otherwise break them up into segments.",
)
@click.option(
    "--overlap",
    "overlap",
    type=float,
    required=False,
    default=0,
    help="Spectrogram overlap in seconds. Default = 0.",
)
@click.option(
    "--power",
    "power",
    type=float,
    required=False,
    help="Raise spectrograms to this power. Lower values show more detail.",
)
def _plot_dir_cmd(
    cfg_path: str,
    ndims: bool,
    input_path: str,
    output_path: str,
    all: bool,
    overlap: float,
    power: float = 1.0,
):
    util.set_logging()
    plot_dir(cfg_path, ndims, input_path, output_path, all, overlap, power)


def plot_rec(
    cfg_path: Optional[str] = None,
    ndims: bool = False,
    input_path: str = "",
    output_path: str = "",
    all: bool = False,
    overlap: float = 0.0,
    power: float = 1.0,
):
    """
    Plot spectrograms for a specific audio recording.

    This command processes a single audio file and generates spectrogram images.
    It can either plot the entire recording as one spectrogram or break it into
    overlapping segments.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - ndims (bool): If True, do not show time and frequency dimensions on the spectrogram plots.
    - input_path (str): Path to the audio recording file to process.
    - output_path (str): Directory where spectrogram images will be saved.
    - all (bool): If True, plot the entire recording as one spectrogram. If False, break into segments.
    - overlap (float): Spectrogram overlap in seconds when breaking the recording into segments. Default is 0.
    - power (float): Raise spectrograms to this power for visualization. Lower values show more detail. Default is 1.0.
    """
    from britekit.core.audio import Audio

    cfg = get_config(cfg_path)
    if power is not None:
        cfg.audio.power = power

    if overlap is None:
        overlap = cfg.infer.overlap

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    audio = Audio(cfg=cfg)
    _plot_recording(cfg, audio, input_path, output_path, all, overlap, ndims)


@click.command(
    name="plot-rec",
    short_help="Plot spectrograms from a specific recording.",
    help=util.cli_help_from_doc(plot_rec.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    required=False,
    help="Path to YAML file defining config overrides.",
)
@click.option(
    "--ndims",
    "ndims",
    is_flag=True,
    help="If specified, do not show seconds on x-axis and frequencies on y-axis.",
)
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to input directory.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory.",
)
@click.option(
    "--all",
    "all",
    is_flag=True,
    help="If specified, plot whole recordings in one spectrogram each. Otherwise break them up into segments.",
)
@click.option(
    "--overlap",
    "overlap",
    type=float,
    required=False,
    default=0,
    help="Spectrogram overlap in seconds. Default = 0.",
)
@click.option(
    "--power",
    "power",
    type=float,
    required=False,
    help="Raise spectrograms to this power. Lower values show more detail.",
)
def _plot_rec_cmd(
    cfg_path: str,
    ndims: bool,
    input_path: str,
    output_path: str,
    all: bool,
    overlap: float,
    power: float = 1.0,
):
    util.set_logging()
    plot_rec(cfg_path, ndims, input_path, output_path, all, overlap, power)
