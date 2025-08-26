import click
import numpy as np
import zlib
from typing import List, Optional

from britekit.core.config_loader import get_config, BaseConfig
from britekit.core.util import expand_spectrogram, get_device, cli_help_from_doc
from britekit.models.model_loader import load_from_checkpoint, BaseModel
from britekit.training_db.training_db import TrainingDatabase


def embed_impl(
    cfg_path: Optional[str],
    db_path: Optional[str],
    class_name: Optional[str],
    spec_group: str,
) -> None:
    """
    Generate embeddings for spectrograms and insert them into the database.

    This command uses a trained model to generate embeddings (feature vectors) for spectrograms
    in the training database. These embeddings can be used for similarity search and other
    downstream tasks. The embeddings are compressed and stored in the database.

    Args:
        cfg_path (str, optional): Path to YAML file defining configuration overrides.
        db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
        class_name (str, optional): Name of a specific class to process. If omitted, processes all classes.
        spec_group (str): Spectrogram group name to process. Defaults to 'default'.
    """

    def embed_block(
        specs: List,
        cfg: BaseConfig,
        model: BaseModel,
        db: TrainingDatabase,
        device: str,
    ) -> None:
        """Process embeddings for a block of spectrograms."""
        spec_array = np.zeros(
            (len(specs), 1, cfg.audio.spec_height, cfg.audio.spec_width)
        )
        value_ids: List[int] = []
        for i, spec in enumerate(specs):
            value_ids.append(spec.specvalue_id)
            spec = expand_spectrogram(spec.value)
            spec = spec.reshape((1, cfg.audio.spec_height, cfg.audio.spec_width))
            spec_array[i] = spec

        embeddings = model.get_embeddings(spec_array, device)
        for i in range(len(embeddings)):
            db.update_specvalue(value_ids[i], "Embedding", zlib.compress(embeddings[i]))

    cfg, _ = get_config(cfg_path)
    if db_path is None:
        db_path = cfg.train.train_db

    BATCH_SIZE = 512  # process this many spectrograms at a time
    model = load_from_checkpoint(cfg.misc.search_ckpt_path)
    device = get_device()
    model.eval()  # set inference mode
    model.to(device)

    with TrainingDatabase(db_path) as db:
        if not class_name:
            results = db.get_class()
            if len(results) == 0:
                click.echo(f"No classes found in database {db_path}.")
                quit()
        else:
            results = db.get_class({"Name": class_name})
            if len(results) == 0:
                click.echo(f"Class {class_name} not found in database {db_path}.")
                quit()

        for result in results:
            click.echo(f"Processing {result.name}")
            specs = db.get_spectrogram_by_class(result.name, spec_group=spec_group)
            click.echo(f"Fetched {len(specs)} spectrograms for {result.name}")
            start_idx = 0

            while start_idx < len(specs):
                end_idx = min(start_idx + BATCH_SIZE, len(specs))
                click.echo(f"Processing spectrograms {start_idx} to {end_idx - 1}")
                embed_block(specs[start_idx:end_idx], cfg, model, db, device)
                start_idx += BATCH_SIZE


@click.command(
    name="embed",
    short_help="Insert spectrogram embeddings into database.",
    help=cli_help_from_doc(embed_impl.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    required=False,
    help="Path to YAML file defining config overrides.",
)
@click.option("-d", "--db", "db_path", required=False, help="Path to the database.")
@click.option("--name", "class_name", required=False, help="Class name")
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    default="default",
    help="Spectrogram group name. Defaults to 'default'.",
)
def embed_cmd(
    cfg_path: Optional[str],
    db_path: Optional[str],
    class_name: Optional[str],
    spec_group: str,
) -> None:
    embed_impl(cfg_path, db_path, class_name, spec_group)
