#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users
# Defer some imports to improve --help performance.
import json
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import click

from britekit.core import util


def _make_session():
    """Reuse connections and back off on transient server/network failures."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _audio_header(content: bytes) -> bool:
    """Reject empty/error pages while accepting common XC audio containers."""
    return (
        content.startswith((b"ID3", b"OggS", b"fLaC"))
        or (len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0)
        or (content.startswith(b"RIFF") and content[8:12] == b"WAVE")
        or content[4:8] == b"ftyp"
    )


def _existing_recordings(output: Path) -> set[str]:
    ids = set()
    for path in output.iterdir():
        match = re.fullmatch(r"XC(\d+)\.mp3", path.name)
        if match and path.is_file():
            with path.open("rb") as audio:
                if _audio_header(audio.read(12)):
                    ids.add(match[1])
    return ids


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

    if int(page) >= int(num_pages):
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
    workers: int = 4,
):
    """
    Download bird song recordings from Xeno-Canto database.

    This command uses the Xeno-Canto API v3 to search for and download audio recordings
    of bird songs. The API requires authentication via an API key. Recordings are
    downloaded as MP3 files and saved to the specified output directory.

    To get an API key, register as a Xeno-Canto user and check your account page.
    Then specify the key in the --key argument, or set the environment variable XCKEY=<key>.

    Args:
    - key (str, optional): Xeno-Canto API key for authentication. Can also be set via XCKEY environment variable.
    - name (str): Required species name to search for (common name or scientific name).
    - output_dir (str): Required directory where downloaded recordings will be saved.
    - max_downloads (int): Maximum total recordings, including existing valid XC files. Default is 500; zero or negative means unlimited.
    - ignore_licence (bool): If True, ignore license restrictions. By default, excludes BY-NC-ND licensed recordings.
    - scientific_name (bool): If True, treat the name as a scientific name rather than common name.
    - seen_only (bool): If True, only download recordings where the animal was seen (animal-seen=yes).
    - workers (int): Concurrent audio downloads. Default is 4; use 1 for sequential downloads.
    """
    import requests

    if workers < 1:
        raise ValueError("workers must be at least 1")

    if key is None:
        if "XCKEY" in os.environ:
            key = os.environ["XCKEY"]
        else:
            logging.error(
                "Xeno-Canto API key must be specified in --key argument or in XCKEY environment variable."
            )
            quit()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    existing = _existing_recordings(output)
    if max_downloads > 0 and len(existing) >= max_downloads:
        logging.info(f"Downloaded 0 recordings ({len(existing)} already present)")
        return

    if scientific_name:
        name = f'sp:"{name.lower()}"'
    else:
        name = f'en:"={name.lower()}"'

    name = quote(name)

    # get list of recordings
    recordings: list = []
    page = 0
    done = False
    with _make_session() as session:
        while not done:
            page += 1
            url = f"https://www.xeno-canto.org/api/3/recordings?query={name}&page={page}&key={key}"
            # Never log the URL: it contains the user's API key.
            logging.info(f"Requesting metadata page {page}")
            try:
                with session.get(url, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    text = response.text
                done = process_response(recordings, text, ignore_licence, seen_only)
            except (requests.RequestException, ValueError, KeyError) as exc:
                logging.error(f"Metadata page {page} failed ({type(exc).__name__})")
                return

    # sort recordings by quality
    recordings = sorted(recordings, key=sort_key)

    # download them
    # De-duplicate API results before scheduling concurrent writes.
    candidates: dict[str, dict] = {}
    for recording in recordings:
        recording_id = str(recording["id"])
        if recording_id.isdigit() and recording_id not in existing:
            candidates.setdefault(recording_id, recording)
    remaining = iter(candidates.values())
    local = threading.local()
    sessions = []
    session_lock = threading.Lock()

    def download(recording) -> bool:
        if not hasattr(local, "session"):
            local.session = _make_session()
            with session_lock:
                sessions.append(local.session)
        outfile = output / f"XC{recording['id']}.mp3"
        logging.info(f"Downloading {outfile}")
        url = recording["file"]
        if not url:
            url = recording["url"]
            if url.startswith("//"):
                url = f"https:{url}"
            url = f"{url.rstrip('/')}/download"
        if url.startswith("//"):
            url = f"https:{url}"
        temp_path = None
        try:
            with local.session.get(url, timeout=(10, 60)) as response:
                response.raise_for_status()
                content = response.content
            if not _audio_header(content[:12]):
                logging.error(f"Invalid audio response for XC{recording['id']}")
                return False
            with tempfile.NamedTemporaryFile(
                dir=output, suffix=".part", delete=False
            ) as audio:
                temp_path = Path(audio.name)
                audio.write(content)
            os.replace(temp_path, outfile)
            return True
        except (requests.RequestException, OSError) as exc:
            logging.error(f"Download XC{recording['id']} failed ({type(exc).__name__})")
            return False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    downloaded = 0
    limit = max_downloads - len(existing) if max_downloads > 0 else None
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending: set[Future[bool]] = set()
            exhausted = False
            while pending or not exhausted:
                while len(pending) < workers and (
                    limit is None or downloaded + len(pending) < limit
                ):
                    recording = next(remaining, None)
                    if recording is None:
                        exhausted = True
                        break
                    pending.add(pool.submit(download, recording))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                downloaded += sum(future.result() for future in finished)
    finally:
        for session in sessions:
            session.close()

    logging.info(f"Downloaded {downloaded} recordings")


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
    help="Maximum total recordings, including existing valid XC files. Default = 500; <=0 is unlimited.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Concurrent audio downloads.",
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
    workers: int,
):
    util.set_logging()
    xeno(
        key,
        name,
        output_dir,
        max_downloads,
        ignore_licence,
        scientific_name,
        seen_only,
        workers,
    )
