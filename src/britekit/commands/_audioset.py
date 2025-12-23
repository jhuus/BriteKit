# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import click

from britekit.core import util


def _download_recording(
    output_dir: str,
    youtube_id: str,
    start_seconds: float,
    sampling_rate: int,
) -> bool:
    # download it as wav, which is faster than downloading as mp3;
    # then convert to mp3 when the 10-second clip is extracted
    from britekit.core.audio import load_audio

    command = f'yt-dlp -q -o "{output_dir}/{youtube_id}.%(EXT)s" -x --audio-format wav https://www.youtube.com/watch?v={youtube_id}'
    logging.info(f"Downloading {youtube_id}")
    os.system(command)

    # extract the 10-second clip and delete the original
    audio_path1 = os.path.join(output_dir, f"{youtube_id}.NA.wav")
    if os.path.exists(audio_path1):
        logging.info("Extracting 10-second clip")
        audio_path2 = os.path.join(output_dir, f"{youtube_id}-{int(start_seconds)}.mp3")
        audio = load_audio(audio_path1, sr=sampling_rate)

        import numpy as np
        import soundfile as sf

        assert isinstance(audio, np.ndarray)
        start_sample = int(start_seconds * sampling_rate)
        end_sample = int((start_seconds + 10) * sampling_rate)
        sf.write(
            audio_path2, audio[start_sample:end_sample], sampling_rate, format="mp3"
        )
        os.remove(audio_path1)
        return True  # succeeded
    else:
        return False  # failed


def _download_class(
    class_name: str,
    output_dir: str,
    max_downloads: int,
    sampling_rate: int,
    num_to_skip: int,
    do_report: bool,
    root_dir: str,
) -> None:
    # read class info
    import pandas as pd

    class_label_path = str(Path(root_dir) / "data" / "audioset" / "class_list.csv")
    df: pd.DataFrame = pd.read_csv(class_label_path)
    name_to_index: Dict[str, int] = {}
    index_to_label: Dict[int, str] = {}
    label_to_name: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        name = str(row.display_name).lower()
        name_to_index[name] = row.index
        index_to_label[row.index] = row.mid
        label_to_name[row.mid] = name

    use_name = class_name.lower()
    if use_name not in name_to_index:
        logging.error(
            f'Class "{class_name}" not found. See names in "{class_label_path}".'
        )
        quit()

    class_index = name_to_index[use_name]
    class_label = index_to_label[class_index]

    # read info for all clips that match the specified class
    logging.info("Scanning unbalanced_train_segments.csv...")
    details_path = str(
        Path(root_dir) / "data" / "audioset" / "unbalanced_train_segments.csv"
    )
    df = pd.read_csv(
        details_path, quotechar='"', skipinitialspace=True, low_memory=False
    )
    label_counts: Dict[str, int] = {}
    num_unique: int = 0  # number with no other labels
    class_rows: List[Tuple[str, float, List[str]]] = []
    for row in df.itertuples(index=False):
        labels = row.positive_labels.split(",")
        if class_label in labels:
            class_rows.append((row.YTID, row.start_seconds, labels))
            if len(labels) == 1:
                num_unique += 1

            for label in labels:
                if label == class_label:
                    continue

                if label not in label_counts:
                    label_counts[label] = 0

                label_counts[label] += 1

    if do_report:
        logging.info(f"# segments with no secondary labels = {num_unique}\n")
        for label in label_counts:
            logging.info(
                f"# segments also labelled {label_to_name[label]} = {label_counts[label]}"
            )

        quit()

    # get any allowable secondary labels
    class_inclusion_path = str(
        Path(root_dir) / "data" / "audioset" / "class_inclusion.csv"
    )
    df = pd.read_csv(class_inclusion_path)
    allowed_labels: Set[str] = set([class_label])
    for i, row in df.iterrows():
        if row["Name"] == class_name:
            for i in range(1, 11):
                label = row[f"Include{i}"]
                if not pd.isna(label):
                    label = label.lower()
                    if label not in name_to_index:
                        logging.error(
                            f'Error: value "{label}" in class_inclusion.csv is not a known class name.'
                        )
                        quit()

                    allowed_labels.add(index_to_label[name_to_index[label]])

    # download recordings and save only the relevant 10-second segment of each
    count = 0
    for youtube_id, start_seconds, labels in class_rows:
        if all(label in allowed_labels for label in labels):
            if count < num_to_skip:
                count += 1
                continue

            if _download_recording(
                output_dir, youtube_id, start_seconds, sampling_rate
            ):
                count += 1
                if count >= max_downloads + num_to_skip:
                    break

    logging.info(f"# downloaded = {count - num_to_skip}")


def _download_curated(
    curated_csv_path: str,
    output_dir: str,
    max_downloads: int,
    sampling_rate: int,
    num_to_skip: int,
) -> None:
    import pandas as pd

    curated = pd.read_csv(curated_csv_path)
    count: int = 0
    for i, row in curated.iterrows():
        if count < num_to_skip:
            count += 1
            continue

        youtube_id = row["YTID"]
        start_seconds = row["start_seconds"]

        if _download_recording(output_dir, youtube_id, start_seconds, sampling_rate):
            count += 1
            if count >= max_downloads + num_to_skip:
                break

    logging.info(f"# downloaded = {count - num_to_skip}")


def audioset(
    class_name: Optional[str] = None,
    curated_csv_path: Optional[str] = None,
    output_dir: str = "",
    max_downloads: int = 500,
    sampling_rate: int = 32000,
    num_to_skip: int = 0,
    do_report: bool = False,
    root_dir: str = ".",
) -> None:
    """
    Download audio recordings from Google AudioSet.

    This command downloads audio clips from Google AudioSet, a large-scale dataset of audio events.
    You can either download a curated set of recordings or search for a specific audio class.
    When using --rpt flag with a class name, it generates a report on associated secondary classes
    instead of downloading recordings.

    Most AudioSet clips contain multiple classes (e.g., "train", "wind", "speech"). The report
    shows which other classes commonly co-occur with the specified class.

    Args:
    - class_name (str): Name of the audio class to download (e.g., "train", "speech", "music").
    - curated_csv_path (str): Path to CSV file containing a curated list of clips to download.
    - output_dir (str): Directory where downloaded recordings will be saved.
    - max_downloads (int): Maximum number of recordings to download. Default is 500.
    - sampling_rate (float): Output sampling rate in Hz. Default is 32000.
    - num_to_skip (int): Number of initial recordings to skip. Default is 0.
    - do_report (bool): If True, generate a report on associated secondary classes instead of downloading.
    - root_dir (str): Directory that contains the data directory. Default is working directory.
    """

    if class_name is None and curated_csv_path is None:
        logging.error("Error. You must specify either --name or --curated.")
        quit()
    elif class_name is not None and curated_csv_path is not None:
        logging.error("Error. You may specify only one of --name or --curated.")
        quit()

    if not do_report and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if class_name:
        _download_class(
            class_name,
            output_dir,
            max_downloads,
            sampling_rate,
            num_to_skip,
            do_report,
            root_dir,
        )
    else:
        assert curated_csv_path is not None
        _download_curated(
            curated_csv_path,
            output_dir,
            max_downloads,
            sampling_rate,
            num_to_skip,
        )


@click.command(
    name="audioset",
    short_help="Download recordings from Google Audioset.",
    help=util.cli_help_from_doc(audioset.__doc__),
)
@click.option("--name", "class_name", type=str, help="Class name.")
@click.option(
    "--curated",
    "curated_csv_path",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to CSV with curated list of clips.",
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
    "--max",
    "max_downloads",
    type=int,
    default=500,
    help="Maximum number of recordings to download. Default = 500.",
)
@click.option(
    "--sr",
    "sampling_rate",
    type=int,
    default=32000,
    help="Output sampling rate (default = 32000).",
)
@click.option(
    "--skip",
    "num_to_skip",
    type=int,
    default=0,
    help="Skip this many initial recordings (default = 0).",
)
@click.option(
    "--rpt",
    "do_report",
    is_flag=True,
    help="Report on secondary classes associated with the specified class.",
)
@click.option(
    "--root",
    "root_dir",
    default=".",
    type=click.Path(file_okay=False),
    help="Root directory containing data directory.",
)
def _audioset_cmd(
    class_name: str,
    curated_csv_path: str,
    output_dir: str,
    max_downloads: int,
    sampling_rate: int,
    num_to_skip: int,
    do_report: bool,
    root_dir: str,
) -> None:
    util.set_logging()
    audioset(
        class_name,
        curated_csv_path,
        output_dir,
        max_downloads,
        sampling_rate,
        num_to_skip,
        do_report,
        root_dir,
    )
