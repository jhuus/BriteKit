#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from typing import Optional
import zlib

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def search(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    class_name: str = "",
    max_dist: float = 0.5,
    exp: float = 0.5,
    num_to_plot: int = 200,
    output_path: str = "",
    input_path: str = "",
    offset: float = 0.0,
    exclude_db: Optional[str] = None,
    class_name2: Optional[str] = None,
    spec_group: str = "default",
):
    """
    Search a database for spectrograms similar to a specified one.

    This command extracts a spectrogram from a given audio file at a specified offset,
    then searches through a database of spectrograms to find the most similar ones
    based on embedding similarity. Results are plotted and saved to the output directory.

    Args:
    - cfg_path (str): Path to YAML configuration file defining model settings.
    - db_path (str): Path to the training database containing spectrograms to search.
    - class_name (str): Name of the class/species to search within the database.
    - max_dist (float): Maximum distance threshold. Results with distance greater than this are excluded.
    - exp (float): Exponent to raise spectrograms to for visualization (shows background sounds).
    - num_to_plot (int): Maximum number of similar spectrograms to plot and save.
    - output_path (str): Directory where search results and plots will be saved.
    - input_path (str): Path to the audio file containing the target spectrogram.
    - offset (float): Time offset in seconds where the target spectrogram is extracted.
    - exclude_db (str, optional): Path to an exclusion database. Spectrograms in this database are excluded from results.
    - class_name2 (str, optional): Class name in the exclusion database. Defaults to the search class name.
    - spec_group (str): Spectrogram group name in the database. Defaults to 'default'.
    """

    class SpecInfo:
        def __init__(self, segment_id, specvalue_id, embedding):
            self.segment_id = segment_id
            self.specvalue_id = specvalue_id
            self.embedding = embedding
            self.distance = 0

    import numpy as np
    import scipy

    from britekit.core.audio import Audio
    from britekit.core.plot import plot_spec
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config(cfg_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get the spectrogram to search for, and plot it
    audio = Audio()
    audio.load(input_path)
    specs, _ = audio.get_spectrograms([offset])
    if specs is None or len(specs) == 0:
        logging.error(
            f"Failed to retrieve search spectrogram from offset {offset} in {input_path}"
        )
        quit()

    target_spec = specs[0].cpu().numpy()
    audio_file_name = os.path.basename(input_path)
    _, ext = os.path.splitext(audio_file_name)
    audio_file_name = audio_file_name[: -(len(ext))]
    image_path = os.path.join(output_path, f"0~{audio_file_name}-{offset:.2f}~0.0.jpeg")
    plot_spec(target_spec**exp, image_path)

    # get spectrograms from the database
    if db_path is None:
        db_path = cfg.train.train_db

    # get recordings and create dict from ID to filename
    logging.info(f"Opening database to search at {db_path}")
    db = TrainingDatabase(db_path)
    recording_dict = {}
    results = db.get_recording_by_class(class_name)

    for r in results:
        recording_dict[r.id] = r.filename

    # get embeddings only, since getting spectrograms here might use too much memory
    results = db.get_spectrogram_by_class(
        class_name, include_value=False, include_embedding=True, spec_group=spec_group
    )
    logging.info(f"Retrieved {len(results)} spectrograms to search")

    spec_infos = []
    for i, r in enumerate(results):
        if r.embedding is None:
            logging.error("Error: not all spectrograms have embeddings")
            quit()
        else:
            embedding = np.frombuffer(zlib.decompress(r.embedding), dtype=np.float32)
            spec_infos.append(SpecInfo(r.segment_id, r.specvalue_id, embedding))

    check_spec_names = {}
    if exclude_db is not None:
        check_db = TrainingDatabase(exclude_db)
        use_name = class_name2 if class_name2 is not None else class_name
        results = check_db.get_spectrogram_by_class(use_name, include_embedding=True)
        for r in results:
            spec_name = f"{r.filename}-{int(round(r.offset))}"
            check_spec_names[spec_name] = 1

    # load the saved model, i.e. the search checkpoint
    if cfg.misc.search_ckpt_path is None:
        logging.error("cfg.misc.search_ckpt_path is not specified")
        return

    if not os.path.exists(cfg.misc.search_ckpt_path):
        logging.error("Invalid value for cfg.misc.search_ckpt_path")
        return

    logging.info("Loading saved models")
    from britekit.core.predictor import Predictor

    predictor = Predictor(cfg.misc.search_ckpt_path)

    # get the embedding for the target spectrogram
    logging.info("Generating embeddings")
    input = np.zeros((1, 1, cfg.audio.spec_height, cfg.audio.spec_width))
    input[0] = target_spec
    embeddings = predictor.get_embeddings(input)
    target_embedding = embeddings[0]

    # compare embeddings and save the distances
    logging.info("Comparing embeddings")
    for i in range(len(spec_infos)):
        spec_infos[i].distance = scipy.spatial.distance.cosine(
            target_embedding, spec_infos[i].embedding
        )

    # sort by distance and plot the results
    logging.info("Sorting results")
    spec_infos = sorted(spec_infos, key=lambda value: value.distance)

    logging.info("Plotting results")
    num_plotted = 0
    spec_num = 0
    for spec_info in spec_infos:
        if num_plotted == num_to_plot or spec_info.distance > max_dist:
            break

        segment_results = db.get_segment({"ID": spec_info.segment_id})
        if len(segment_results) != 1:
            logging.error(f"Error: unable to retrieve segment {spec_info.segment_id}")

        value_results = db.get_specvalue({"ID": spec_info.specvalue_id})
        if len(value_results) != 1:
            logging.error(
                f"Error: unable to retrieve spectrogram value {spec_info.specvalue_id}"
            )

        filename = recording_dict[segment_results[0].recording_id]
        offset = segment_results[0].offset
        distance = spec_info.distance
        spec = util.expand_spectrogram(value_results[0].value)
        spec = spec.reshape((1, cfg.audio.spec_height, cfg.audio.spec_width))

        spec_name = f"{filename}-{int(round(offset))}"
        if spec_name in check_spec_names:
            continue

        spec_num += 1
        base, ext = os.path.splitext(filename)
        spec_path = os.path.join(
            output_path, f"{spec_num}~{base}-{offset:.2f}~{distance:.3f}.jpeg"
        )

        if not os.path.exists(spec_path):
            spec **= exp
            plot_spec(spec, spec_path)
            num_plotted += 1


@click.command(
    name="search",
    short_help="Search a database for spectrograms similar to one given.",
    help=util.cli_help_from_doc(search.__doc__),
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
    "-d",
    "--db",
    "db_path",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to the database. Defaults to value of cfg.train.training_db.",
)
@click.option("--name", "class_name", type=str, required=True, help="Class name")
@click.option(
    "--dist",
    "max_dist",
    type=float,
    default=0.5,
    help="Exclude results with distance greater than this. Default = .5.",
)
@click.option(
    "--exp",
    type=float,
    default=0.5,
    help="Raise spectrograms to this exponent to show background sounds. Default = .5.",
)
@click.option(
    "--num",
    "num_to_plot",
    type=int,
    default=200,
    help="Only plot up to this many spectrograms. Default = 200.",
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
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to recording containing spectrogram to search for.",
)
@click.option(
    "--offset",
    type=float,
    default=0.0,
    help="Offset in seconds of spectrogram to search for. Default = 0.",
)
@click.option(
    "-x",
    "--exclude",
    "exclude_db",
    type=click.Path(file_okay=True, dir_okay=False),
    help="If specified, exclude spectrograms that exist in this database.",
)
@click.option(
    "--name2",
    "class_name2",
    type=str,
    help="If --exclude is specified, this is class name in exclude database. Default is the search class name.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    default="default",
    help="Spectrogram group name. Defaults to 'default'.",
)
def _search_cmd(
    cfg_path: str,
    db_path: str,
    class_name: str,
    max_dist: float,
    exp: float,
    num_to_plot: int,
    output_path: str,
    input_path: str,
    offset: float,
    exclude_db: str,
    class_name2: str,
    spec_group: str,
):
    util.set_logging()
    search(
        cfg_path,
        db_path,
        class_name,
        max_dist,
        exp,
        num_to_plot,
        output_path,
        input_path,
        offset,
        exclude_db,
        class_name2,
        spec_group,
    )
