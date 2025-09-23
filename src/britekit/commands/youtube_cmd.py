import os

import click
import numpy as np
import librosa
import soundfile as sf

from britekit.core.util import cli_help_from_doc


def youtube(
    youtube_id: str,
    output_dir: str,
    sampling_rate: int,
) -> None:
    """
    Download an audio recording from Youtube, given a Youtube ID.

    Args:
        youtube_id (str): ID of the clip to download.
        output_dir (str): Directory where downloaded recordings will be saved.
        sampling_rate (float): Output sampling rate in Hz. Default is 32000.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # download it as wav, which is faster than downloading as mp3;
    # then resample and convert to mp3
    command = f'yt-dlp -q -o "{output_dir}/{youtube_id}.%(EXT)s" -x --audio-format wav https://www.youtube.com/watch?v={youtube_id}'
    click.echo(f"Downloading {youtube_id}")
    os.system(command)

    # resample and delete the original
    audio_path1 = os.path.join(output_dir, f"{youtube_id}.NA.wav")
    if os.path.exists(audio_path1):
        audio_path2 = os.path.join(output_dir, f"{youtube_id}.mp3")
        audio, sr = librosa.load(audio_path1, sr=sampling_rate)
        assert isinstance(sr, int)
        assert isinstance(audio, np.ndarray)
        sf.write(audio_path2, audio, sr, format="mp3")
        os.remove(audio_path1)
    else:
        click.echo("Download failed")


@click.command(
    name="youtube",
    short_help="Download a recording from Youtube.",
    help=cli_help_from_doc(youtube.__doc__),
)
@click.option(
    "--id",
    "youtube_id",
    required=True,
    type=str,
    help="Youtube ID.",
)
@click.option(
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
    youtube(
        youtube_id,
        output_dir,
        sampling_rate,
    )
