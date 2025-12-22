# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os

import click

from britekit.core import util


def youtube(
    id: str = "",
    output_dir: str = "",
    sampling_rate: int = 32000,
) -> None:
    """
    Download an audio recording from Youtube, given a Youtube ID.

    Args:
    - id (str): ID of the clip to download.
    - output_dir (str): Directory where downloaded recordings will be saved.
    - sampling_rate (float): Output sampling rate in Hz. Default is 32000.
    """
    import numpy as np
    import soundfile as sf
    from britekit.core.audio import load_audio

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # download it as wav, which is faster than downloading as mp3;
    # then resample and convert to mp3
    command = f'yt-dlp -q -o "{output_dir}/{id}.%(EXT)s" -x --audio-format wav https://www.youtube.com/watch?v={id}'
    logging.info(f"Downloading {id}")
    os.system(command)

    # resample and delete the original
    audio_path1 = os.path.join(output_dir, f"{id}.NA.wav")
    if os.path.exists(audio_path1):
        audio_path2 = os.path.join(output_dir, f"{id}.mp3")
        audio = load_audio(audio_path1, sr=sampling_rate)
        assert isinstance(audio, np.ndarray)
        sf.write(audio_path2, audio, sampling_rate, format="mp3")
        os.remove(audio_path1)
    else:
        logging.info("Download failed")


@click.command(
    name="youtube",
    short_help="Download a recording from Youtube.",
    help=util.cli_help_from_doc(youtube.__doc__),
)
@click.option(
    "--id",
    "youtube_id",
    required=True,
    type=str,
    help="Youtube ID.",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory.",
)
@click.option(
    "--sr",
    "sampling_rate",
    type=int,
    default=32000,
    help="Output sampling rate (default = 32000).",
)
def _youtube_cmd(
    youtube_id: str,
    output_dir: str,
    sampling_rate: int,
) -> None:
    util.set_logging()
    youtube(
        youtube_id,
        output_dir,
        sampling_rate,
    )
