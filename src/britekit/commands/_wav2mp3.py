#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users
import logging
import os

import click

from britekit.core import util


def wav2mp3(
    dir: str = "",
    sampling_rate: int = 32000,
):
    """
    Convert uncompressed audio files to MP3 format and replace the originals.

    This command processes all uncompressed audio files in a directory and converts
    them to MP3 format using FFmpeg. Supported input formats include FLAC, WAV, WMA,
    AIFF, and other uncompressed audio formats. After successful conversion, the
    original files are deleted to save disk space.

    The conversion uses a 192k bitrate for good quality while maintaining reasonable
    file sizes. This is useful for standardizing audio formats and reducing storage
    requirements for large audio datasets.

    Args:
    - dir (str): Path to directory containing audio files to convert.
    - sampling_rate (int): Output sampling rate in Hz. Default is 32000 Hz.
    """
    CONVERT_TYPES = {
        ".flac",
        ".octet-stream",
        ".qt",
        ".wav",
        ".wma",
        ".x-hx-aac-adts",
        ".x-aiff",
    }

    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            base, ext = os.path.splitext(filename)
            if ext and (ext.lower() in CONVERT_TYPES):
                target = os.path.join(dir, base)
                cmd = f'ffmpeg -i "{filepath}" -y -vn -ar {sampling_rate} -b:a 192k "{target}.mp3"'
                logging.info(cmd)
                os.system(cmd)
                os.remove(filepath)


@click.command(
    name="wav2mp3",
    short_help="Convert uncompressed audio or flac to mp3.",
    help=util.cli_help_from_doc(wav2mp3.__doc__),
)
@click.option(
    "--dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to directory containing recordings.",
)
@click.option(
    "--sr",
    "sampling_rate",
    type=int,
    default=32000,
    help="Output sampling rate (default = 32000).",
)
def _wav2mp3_cmd(
    dir: str,
    sampling_rate: int,
):
    util.set_logging()
    wav2mp3(dir, sampling_rate)
