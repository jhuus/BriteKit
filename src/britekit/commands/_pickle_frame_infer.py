"""Generate the pickle-frame format directly from an SED ensemble."""

from typing import Optional

import click

from britekit.core import util
from ._teacher_targets import _generate_targets


def pickle_frame_infer(
    train_pickle_path: str,
    checkpoint_path: str,
    output_path: str,
    cfg_path: Optional[str] = None,
    batch_size: int = 256,
) -> None:
    """Generate soft frame labels using a checkpoint or ensemble.

    Read the spectrograms, segment IDs and class labels from a training pickle.
    For each segment, average the checkpoints' uncalibrated frame probabilities
    for its labeled class. Save the same dictionary as pickle-frame:
    {segment_id: float32 array of shape (num_frames,)}. No thresholding, padding,
    normalization by peak score, or filling of gaps is applied.

    Every segment must have exactly one class label. Checkpoints must provide
    SED frame outputs, cover the training classes, and share the source frontend,
    segment duration and sed_fps. Class order may differ; classes are matched by
    code. Each checkpoint supplies its own linear-to-dB preprocessing settings.
    The training pickle must contain features from the same frontend used to
    train the checkpoints; it does not contain metadata to verify this fully.

    Set train.frame_label_pickle to the output file for subsequent training,
    with audio.spec_duration and train.sed_fps matching the checkpoints. Existing
    segment labels are retained. This does not require teacher_targets_pickle.
    The inference device is selected automatically, as for analyze.

    Args:
    - train_pickle_path (str): Training pickle containing spectrograms and stable segment IDs.
    - checkpoint_path (str): SED checkpoint or directory of ensemble checkpoints.
    - output_path (str): Output frame-label pickle, compatible with pickle-frame.
    - cfg_path (str, optional): YAML configuration overrides; checkpoint audio settings take precedence.
    - batch_size (int): Number of spectrograms per inference batch.
    """
    _generate_targets(
        train_pickle_path,
        checkpoint_path,
        output_path,
        cfg_path,
        batch_size,
        device=None,
        frame_labels_only=True,
    )


@click.command(
    name="pickle-frame-infer",
    short_help="Generate a soft frame-label pickle using an SED ensemble.",
    help=util.cli_help_from_doc(pickle_frame_infer.__doc__),
)
@click.argument("train_pickle_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--checkpoints",
    "checkpoint_path",
    type=click.Path(exists=True),
    required=True,
    help="SED checkpoint or directory containing an ensemble.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Output frame-label pickle for train.frame_label_pickle.",
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML overrides; checkpoint audio settings take precedence.",
)
@click.option(
    "--batch-size", type=click.IntRange(min=1), default=256, show_default=True
)
def _pickle_frame_infer_cmd(**kwargs) -> None:
    util.set_logging()
    pickle_frame_infer(**kwargs)
