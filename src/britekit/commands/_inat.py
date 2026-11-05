#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import csv
import logging
import os
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import click

from britekit.core import util
from britekit.core.http_util import create_download_session, is_audio_header


def _paths(
    url: Optional[str], output_dir: str, no_prefix: bool
) -> Optional[tuple[str, Path, Path]]:
    if not url or not url.strip():
        return None
    filename = urlsplit(url).path.rsplit("/", 1)[-1]
    base, _ = os.path.splitext(filename)
    if not base or filename in (".", ".."):
        return None
    prefix = "" if no_prefix else "N"
    return (
        base,
        Path(output_dir) / f"{prefix}{filename}",
        Path(output_dir) / f"{prefix}{base}.mp3",
    )


def _valid_audio(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as audio:
        return is_audio_header(audio.read(12))


def _download(
    url: Optional[str], output_dir: str, no_prefix: bool, session=None
) -> Optional[str]:
    import requests

    paths = _paths(url, output_dir, no_prefix)
    if paths is None:
        return None
    base, output_path, mp3_path = paths
    # Also recognize a previously downloaded file converted to MP3.
    if _valid_audio(output_path) or _valid_audio(mp3_path):
        return base
    own_session = session is None
    if own_session:
        session = create_download_session()
    temp_path = None
    try:
        logging.info(f"Downloading {output_path}")
        with session.get(url, allow_redirects=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            content = response.content
        if not is_audio_header(content[:12]):
            logging.error(f"Invalid audio response for {output_path.name}")
            return None
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".part", delete=False
        ) as audio:
            temp_path = Path(audio.name)
            audio.write(content)
        os.replace(temp_path, output_path)
        return base
    except (requests.RequestException, OSError) as exc:
        logging.error(f"Download {output_path.name} failed ({type(exc).__name__})")
        return None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if own_session:
            session.close()


def inat(
    name: str = "",
    output_dir: str = "",
    max_downloads: int = 500,
    no_prefix: bool = False,
    include_unverified: bool = False,
    workers: int = 4,
) -> None:
    """
    Download audio recordings from iNaturalist observations.

    This command searches iNaturalist for observations of a specified species that contain
    audio recordings. It downloads the audio files and creates a CSV file mapping the
    downloaded files to their iNaturalist observation URLs for reference.

    Observations with "needs_id" quality are excluded unless explicitly requested.
    The command respects the maximum download limit and can optionally add filename prefixes.

    Args:
    - name (str): Required species name to search for (e.g., "Common Yellowthroat", "Geothlypis trichas").
    - output_dir (str): Required directory where downloaded recordings will be saved.
    - max_downloads (int): Maximum total matching recordings, including existing valid files. Default is 500; zero or negative means unlimited.
    - no_prefix (bool): If True, skip adding "N" prefix to filenames. Default adds prefix.
    - include_unverified (bool): If true, include recordings that have not been verified. By default they are excluded.
    - workers (int): Concurrent audio downloads. Default is 4; use 1 for sequential downloads.
    """
    import pyinaturalist

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    response: Dict[str, Any] = pyinaturalist.get_observations(
        taxon_name=f"{name}", sounds=True, page="all", per_page=200
    )

    # Keep references from previous runs, including recordings converted to MP3.
    csv_path = Path(output_dir) / "inat.csv"
    id_map: Dict[str, str] = {}
    if csv_path.exists():
        with csv_path.open(newline="") as csv_file:
            id_map.update(
                (row["Media ID"], row["URL"]) for row in csv.DictReader(csv_file)
            )
    logging.info(f"Response contains {len(response['results'])} results")
    candidates: Dict[str, tuple[str, int]] = {}
    for i, result in enumerate(response["results"]):
        for sound in result["sounds"]:
            url = sound["file_url"]
            if not url:
                logging.info(f"Skipping {sound['file_url']} (no audio available)")
                continue

            if result["quality_grade"] == "needs_id":
                if include_unverified:
                    logging.info(
                        f"Warning: result {i + 1} has not been verified ({sound['file_url']})"
                    )
                else:
                    logging.info(
                        f"Skipping result {i + 1}, since it has not been verified ({sound['file_url']})"
                    )
                    continue

            paths = _paths(url, output_dir, no_prefix)
            if paths is not None and result["id"] is not None:
                candidates.setdefault(paths[0], (url, result["id"]))

    existing = 0
    todo = []
    for media_id, (url, observation_id) in candidates.items():
        paths = _paths(url, output_dir, no_prefix)
        assert paths is not None  # Candidates were validated above.
        _, output_path, mp3_path = paths
        if _valid_audio(output_path) or _valid_audio(mp3_path):
            existing += 1
            id_map[media_id] = (
                f"https://www.inaturalist.org/observations/{observation_id}"
            )
        else:
            todo.append((url, observation_id))

    local = threading.local()
    sessions = []
    session_lock = threading.Lock()

    def download(url: str) -> Optional[str]:
        if not hasattr(local, "session"):
            local.session = create_download_session()
            with session_lock:
                sessions.append(local.session)
        return _download(url, output_dir, no_prefix, session=local.session)

    remaining = iter(todo)
    limit = max(0, max_downloads - existing) if max_downloads > 0 else None
    num_downloads = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending: Dict[Future[Optional[str]], int] = {}
            exhausted = False
            while pending or not exhausted:
                while len(pending) < workers and (
                    limit is None or num_downloads + len(pending) < limit
                ):
                    item = next(remaining, None)
                    if item is None:
                        exhausted = True
                        break
                    url, observation_id = item
                    pending[pool.submit(download, url)] = observation_id
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    observation_id = pending.pop(future)
                    downloaded_id = future.result()
                    if downloaded_id is not None:
                        num_downloads += 1
                        id_map[downloaded_id] = (
                            f"https://www.inaturalist.org/observations/{observation_id}"
                        )
    finally:
        for session in sessions:
            session.close()

    logging.info(f"Downloaded {num_downloads} recordings")
    if not id_map:
        return

    # Replace the mapping atomically so a failed write cannot truncate prior references.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", dir=output_dir, suffix=".part", delete=False
        ) as csv_temp:
            temp_path = Path(csv_temp.name)
            writer = csv.writer(csv_temp)
            writer.writerow(["Media ID", "URL"])
            writer.writerows((key, id_map[key]) for key in sorted(id_map))
        os.replace(temp_path, csv_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


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
    help="Maximum total matching recordings, including existing valid files. Default = 500; <=0 is unlimited.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Concurrent audio downloads.",
)
@click.option(
    "--noprefix",
    "no_prefix",
    is_flag=True,
    help="By default, filenames use an 'N' prefix and recording number. Specify this flag to skip the prefix.",
)
@click.option(
    "--nover",
    "include_unverified",
    is_flag=True,
    help="If specified, include recordings that have not been verified. By default they are excluded.",
)
def _inat_cmd(
    name: str,
    output_dir: str,
    max_downloads: int,
    no_prefix: bool,
    include_unverified: bool,
    workers: int,
) -> None:
    util.set_logging()
    inat(name, output_dir, max_downloads, no_prefix, include_unverified, workers)
