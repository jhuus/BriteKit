# File name starts with _ to keep it out of typeahead for API users
# Defer some imports to improve --help performance.
import json
import logging
import os
from typing import Optional
from urllib.parse import quote

import click

from britekit.core import util


def sort_key(recording):
    """Sort recordings by quality. If no quality given, sort between A and B."""
    quality = recording["q"]

    if quality == "A":
        return 1
    elif quality == "":
        return 2
    elif quality == "B":
        return 3
    elif quality == "C":
        return 4
    elif quality == "D":
        return 5
    else:
        return 6


def process_response(
    recordings: list, text: str, ignore_licence: bool, seen_only: bool
):
    """Process a response from the Xeno-Canto API."""

    j = json.loads(text)
    page = j["page"]
    num_pages = j["numPages"]
    curr_recordings = j["recordings"]

    logging.info(f"Response contains {len(curr_recordings)} recordings.")
    for recording in curr_recordings:
        if not ignore_licence and "by-nc-nd" in recording["lic"]:
            continue

        if seen_only and recording["animal-seen"] != "yes":
            continue

        recordings.append(recording)

    if page == num_pages:
        return True  # done
    else:
        return False  # not done


def xeno(
    key: Optional[str] = None,
    name: str = "",
    output_dir: str = "",
    max_downloads: int = 500,
    ignore_licence: bool = False,
    scientific_name: bool = False,
    seen_only: bool = False,
):
    """
    Download bird song recordings from Xeno-Canto database.

    This command uses the Xeno-Canto API v3 to search for and download audio recordings
    of bird songs. The API requires authentication via an API key. Recordings are
    downloaded as MP3 files and saved to the specified output directory.

    To get an API key, register as a Xeno-Canto user and check your account page.
    Then specify the key in the --key argument, or set the environment variable XCKEY=<key>.

    Args:
    - key (str): Xeno-Canto API key for authentication. Can also be set via XCKEY environment variable.
    - output_dir (str): Directory where downloaded recordings will be saved.
    - max_downloads (int): Maximum number of recordings to download. Default is 500.
    - name (str): Species name to search for (common name or scientific name).
    - ignore_licence (bool): If True, ignore license restrictions. By default, excludes BY-NC-ND licensed recordings.
    - scientific_name (bool): If True, treat the name as a scientific name rather than common name.
    - seen_only (bool): If True, only download recordings where the animal was seen (animal-seen=yes).
    """
    import requests

    if key is None:
        if "XCKEY" in os.environ:
            key = os.environ["XCKEY"]
        else:
            logging.error(
                "Xeno-Canto API key must be specified in --key argument or in XCKEY environment variable."
            )
            quit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if scientific_name:
        name = f'sp:"{name.lower()}"'
    else:
        name = f'en:"={name.lower()}"'

    name = quote(name)

    # get list of recordings
    recordings: list = []
    page = 0
    done = False
    while not done:
        page += 1

        url = f"https://www.xeno-canto.org/api/3/recordings?query={name}&page={page}&key={key}"
        logging.info(f"Requesting data from {url}")
        response = requests.get(url)

        # try up to 3 times if status=503 (server temporarily unavailable)
        for i in range(3):
            if response.status_code == 200:
                done = process_response(
                    recordings, response.text, ignore_licence, seen_only
                )
                break
            elif response.status_code == 503 and i < 2:
                logging.error(
                    f"HTTP GET returned status={response.status_code}. Retrying."
                )
            else:
                logging.error(f"HTTP GET failed with status={response.status_code}")
                done = True

    # sort recordings by quality
    recordings = sorted(recordings, key=sort_key)

    # download them
    downloaded = 0
    for recording in recordings:
        outfile = os.path.join(output_dir, f"XC{recording['id']}.mp3")
        if not os.path.exists(outfile):
            logging.info(f"Downloading {outfile}")
            url = recording["file"]
            if not url:
                url = f"https:{recording['url']}/download"

            response = requests.get(url)
            with open(outfile, "wb") as mp3:
                mp3.write(response.content)
                downloaded += 1

                if max_downloads > 0 and downloaded >= max_downloads:
                    return


@click.command(
    name="xeno",
    short_help="Download recordings from Xeno-Canto.",
    help=util.cli_help_from_doc(xeno.__doc__),
)
@click.option("--key", type=str, help="Xeno-Canto API key.")
@click.option("--name", required=True, type=str, help="Species name.")
@click.option(
    "-o",
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory.",
)
@click.option(
    "--max",
    "max_downloads",
    type=int,
    default=500,
    help="Maximum number of recordings to download. Default = 500.",
)
@click.option(
    "--nolic",
    "ignore_licence",
    is_flag=True,
    help="Specify this flag to ignore the licence. By default, exclude if licence is BY-NC-ND.",
)
@click.option(
    "--sci",
    "scientific_name",
    is_flag=True,
    help="Specify this flag when using a scientific name rather than a common name.",
)
@click.option(
    "--seen",
    "seen_only",
    is_flag=True,
    help="Specify this flag to download only if animal-seen=yes.",
)
def _xeno_cmd(
    key: str,
    name: str,
    output_dir: str,
    max_downloads: int,
    ignore_licence: bool,
    scientific_name: bool,
    seen_only: bool,
):
    util.set_logging()
    xeno(
        key, name, output_dir, max_downloads, ignore_licence, scientific_name, seen_only
    )
