#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from typing import Any, Dict, Optional

import click

from britekit.core import util


def _download(url: Optional[str], output_dir: str, no_prefix: bool) -> Optional[str]:
    import requests

    if url is None or len(url.strip()) == 0:
        return None

    tokens = url.split("?")
    tokens2 = tokens[0].split("/")
    filename = tokens2[-1]

    base, _ = os.path.splitext(filename)

    # check mp3_path too in case file was converted to mp3
    if no_prefix:
        output_path = f"{output_dir}/{filename}"
        mp3_path = f"{output_dir}/{base}.mp3"
    else:
        output_path = f"{output_dir}/N{filename}"
        mp3_path = f"{output_dir}/N{base}.mp3"

    if not os.path.exists(output_path) and not os.path.exists(mp3_path):
        logging.info(f"Downloading {output_path}")
        r = requests.get(url, allow_redirects=True)
        open(output_path, "wb").write(r.content)

    return base


def inat(
    name: str = "",
    output_dir: str = "",
    max_downloads: int = 500,
    no_prefix: bool = False,
) -> None:
    """
    Download audio recordings from iNaturalist observations.

    This command searches iNaturalist for observations of a specified species that contain
    audio recordings. It downloads the audio files and creates a CSV file mapping the
    downloaded files to their iNaturalist observation URLs for reference.

    Only observations with "research grade" quality are downloaded (excluding "needs_id").
    The command respects the maximum download limit and can optionally add filename prefixes.

    Args:
    - output_dir (str): Directory where downloaded recordings will be saved.
    - max_downloads (int): Maximum number of recordings to download. Default is 500.
    - name (str): Species name to search for (e.g., "Common Yellowthroat", "Geothlypis trichas").
    - no_prefix (bool): If True, skip adding "N" prefix to filenames. Default adds prefix.
    """
    import pyinaturalist

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    response: Dict[str, Any] = pyinaturalist.get_observations(
        taxon_name=f"{name}", identified=True, sounds=True, photos=False, page="all"
    )

    id_map: Dict[str, int] = {}  # map media IDs to observation IDs
    logging.info(f"Response contains {len(response['results'])} results")
    num_downloads = 0
    for result in response["results"]:
        if num_downloads >= max_downloads:
            break

        if result["quality_grade"] == "needs_id":
            continue

        for sound in result["sounds"]:
            if sound["file_url"] is None:
                continue

            media_id = _download(sound["file_url"], output_dir, no_prefix)
            if media_id is not None and result["id"] is not None:
                num_downloads += 1
                id_map[media_id] = result["id"]

    csv_path = os.path.join(output_dir, "inat.csv")
    with open(csv_path, "w") as csv_file:
        csv_file.write("Media ID,URL\n")
        for key in sorted(id_map.keys()):
            csv_file.write(
                f"{key},https://www.inaturalist.org/observations/{id_map[key]}\n"
            )


@click.command(
    name="inat",
    short_help="Download recordings from iNaturalist.",
    help=util.cli_help_from_doc(inat.__doc__),
)
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
    "--noprefix",
    "no_prefix",
    is_flag=True,
    help="By default, filenames use an 'N' prefix and recording number. Specify this flag to skip the prefix.",
)
def _inat_cmd(
    name: str,
    output_dir: str,
    max_downloads: int,
    no_prefix: bool,
) -> None:
    util.set_logging()
    inat(name, output_dir, max_downloads, no_prefix)
