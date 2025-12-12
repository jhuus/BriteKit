from __future__ import annotations
from dataclasses import is_dataclass, asdict
from enum import Enum
import glob
import inspect
import logging
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, cast, Dict, List, Union, Mapping, TypeAlias, Optional, Tuple
from posixpath import splitext
import zlib

from britekit.core.config_loader import get_config
from britekit.core.exceptions import InputError

# =============================================================================
# Constants and Type Definitions
# =============================================================================

JSONValue: TypeAlias = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]

AUDIO_EXTS = [
    ".3gp",
    ".3gpp",
    ".8svx",
    ".aa",
    ".aac",
    ".aax",
    ".act",
    ".aif",
    ".aiff",
    ".alac",
    ".amr",
    ".ape",
    ".au",
    ".awb",
    ".cda",
    ".dss",
    ".dvf",
    ".flac",
    ".gsm",
    ".iklax",
    ".ivs",
    ".m4a",
    ".m4b",
    ".m4p",
    ".mmf",
    ".mp3",
    ".mpc",
    ".mpga",
    ".msv",
    ".nmf",
    ".octet-stream",
    ".ogg",
    ".oga",
    ".mogg",
    ".opus",
    ".org",
    ".ra",
    ".rm",
    ".raw",
    ".rf64",
    ".sln",
    ".tta",
    ".voc",
    ".vox",
    ".wav",
    ".wma",
    ".wv",
    ".webm",
    ".x-m4a",
]

_ARG_HEADINGS = ("Args", "Arguments", "Parameters")

# =============================================================================
# Core Utility Functions
# =============================================================================


def format_elapsed_time(start_time: float, end_time: float) -> str:
    """Format elapsed time as HH:MM:SS or MM:SS."""
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time")

    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    if hours == 0:
        return f"{minutes:02}:{seconds:02}"
    else:
        return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_device() -> str:
    """Return the device for pytorch to use."""
    import torch

    cfg = get_config()
    if cfg.misc.force_cpu:
        return "cpu"  # for performance comparisons
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_range(min_val: float, max_val: float, incr: float) -> List[float]:
    """
    Return list of floats from min_val to max_val by incr.
    This addresses issues with np.arange(min, max + increment, increment), which
    may introduce rounding errors that cause the range to slightly overshoot max.
    """
    import numpy as np

    if incr == 0:
        # avoid divide-by-zero
        return [min_val]

    if max_val < min_val:
        raise ValueError("max_val must be greater than or equal to min_val")

    if incr < 0:
        raise ValueError("increment must be positive")

    n_steps = int(np.round((max_val - min_val) / incr))
    values = np.round(np.linspace(min_val, min_val + n_steps * incr, n_steps + 1), 10)
    return [float(v) for v in values]


def get_seconds_from_time_string(time_str: str) -> int:
    """
    Convert a time string into an integer number of seconds.

    Supports the following formats:
    - "71" → 71 seconds
    - "1:11" → 71 seconds
    - "0:01:11" → 71 seconds
    - "1:02:03" → 3723 seconds (1 hour, 2 minutes, 3 seconds)

    Args:
        time_str (str): Time string in seconds or colon-separated format.

    Returns:
        int: Total number of seconds.
    """
    if time_str is None:
        return 0

    time_str = time_str.strip()
    if len(time_str) == 0:
        return 0

    parts = time_str.split(":")

    # Only seconds provided
    if len(parts) == 1:
        return int(float(parts[0]))

    # Minutes and seconds
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        return int(minutes * 60 + seconds)

    # Hours, minutes, and seconds
    elif len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return int(hours * 3600 + minutes * 60 + seconds)

    else:
        raise ValueError(f"Unrecognized time format: '{time_str}'")


def set_logging(level=logging.INFO, timestamp=False):
    """Initialize logging."""
    if timestamp:
        logging.basicConfig(
            stream=sys.stderr,
            level=level,
            format="%(asctime)s.%(msecs)03d %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
    else:
        logging.basicConfig(
            stream=sys.stderr, level=level, format="%(message)s", force=True
        )


# =============================================================================
# Configuration Functions
# =============================================================================


def cfg_to_pure(obj: Any) -> JSONValue:
    """
    Convert complex configuration objects to JSON-serializable format.

    Configuration objects often contain complex types (dataclasses, OmegaConf objects,
    numpy arrays, torch tensors, etc.) that cannot be directly serialized to JSON.
    This function recursively converts these objects to basic Python types (dict, list,
    str, int, float, bool) that can be safely serialized.

    Args:
    - obj: Any object to convert to JSON-serializable format

    Returns:
        JSON-serializable representation of the input object
    """

    # Prevent infinite recursion with a simple depth limit
    def _cfg_to_pure_recursive(obj: Any, depth: int = 0) -> JSONValue:
        if depth > 100:  # Prevent infinite recursion
            return str(obj)

        # Only dataclass *instances*
        if is_dataclass(obj) and not isinstance(obj, type):
            d = asdict(cast(Any, obj))  # cast narrows away the class branch for MyPy
            return {k: _cfg_to_pure_recursive(v, depth + 1) for k, v in d.items()}

        # OmegaConf DictConfig / ListConfig
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore

            if isinstance(obj, (DictConfig, ListConfig)):
                return _cfg_to_pure_recursive(
                    OmegaConf.to_container(obj, resolve=True), depth + 1
                )
        except Exception:
            pass

        # mappings
        if isinstance(obj, Mapping):
            return {
                str(k): _cfg_to_pure_recursive(v, depth + 1) for k, v in obj.items()
            }

        # sequences
        if isinstance(obj, (list, tuple, set)):
            return [_cfg_to_pure_recursive(v, depth + 1) for v in obj]

        # pathlib, enums
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value

        # numpy
        try:
            import numpy as np  # type: ignore

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
        except Exception:
            pass

        # torch odds and ends
        try:
            import torch

            if isinstance(obj, torch.dtype) or isinstance(obj, torch.device):
                return str(obj)
            if isinstance(obj, torch.Size):
                return list(obj)
            if isinstance(obj, torch.Tensor):
                return {"tensor": {"shape": list(obj.shape), "dtype": str(obj.dtype)}}
        except Exception:
            pass

        # callables/classes -> qualified name (avoids __name__ issues)
        if isinstance(obj, type) or callable(obj):
            try:
                mod = getattr(obj, "__module__", "")
                qn = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
                return f"{mod}.{qn}" if mod else qn
            except Exception:
                return str(obj)

        # primitives already fine
        return obj  # type: ignore[return-value]

    return _cfg_to_pure_recursive(obj)


# =============================================================================
# Documentation/CLI Functions
# =============================================================================


def cli_help_from_doc(doc: str | None) -> str | None:
    """
    Return a docstring with everything up to (but not including) the args section.
    Leaves the original doc unchanged for API docs.
    """
    if not doc:
        return None
    text = inspect.cleandoc(doc)
    # Cut at the start of the first args-like section heading.
    m = re.search(
        rf'^\s*(?:{"|".join(_ARG_HEADINGS)})\s*:\s*$',
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Fix: Safe handling of regex match
    if m is not None:
        return text[: m.start()].rstrip()
    else:
        return text


# =============================================================================
# File System Functions
# =============================================================================


def get_audio_files(path: str, short_names: bool = False) -> List[str]:
    """
    Return list of audio files in the given directory.

    Args:
    - path (str): Directory path
    - short_names (bool): If true, return file names, else return full paths

    Returns:
        List of audio files in the given directory
    """
    if not path:
        return []

    files = []
    try:
        if os.path.isdir(path):
            for file_name in sorted(os.listdir(path)):
                file_path = os.path.join(path, file_name)
                # Fix: Check if file still exists before processing
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file_path)
                    if ext is not None and len(ext) > 0 and ext.lower() in AUDIO_EXTS:
                        if short_names:
                            files.append(file_name)
                        else:
                            # convert relative path to absolute path
                            try:
                                files.append(str(Path(file_path).resolve()))
                            except (OSError, RuntimeError):
                                # Fallback to original path if resolve fails
                                files.append(file_path)
    except (OSError, PermissionError) as e:
        logging.error(f"Error accessing directory {path}: {e}")
        return []

    return sorted(files)


def get_file_lines(path: str, encoding: str = "utf-8") -> List[str]:
    """
    Return list of strings representing the lines in a text file,
    removing leading and trailing whitespace and ignoring blank lines
    and lines that start with #.

    Args:
    - path: Path to text file
    - encoding: File encoding (default: utf-8)

    Returns:
        List of lines
    """
    if not path:
        return []

    try:
        with open(path, "r", encoding=encoding) as file:
            lines = []
            for line in file.readlines():
                line = line.strip()
                # Fix: Safe string indexing
                if line and not line.startswith("#"):
                    lines.append(line)

            return lines
    except (IOError, UnicodeDecodeError) as e:
        logging.error(f"Unable to open input file {path}: {e}")
        return []


def get_source_name(filename: str) -> str:
    """
    Return a source name given a recording file name.

    Args:
    - filename: Recording file name

    Returns:
        Source name
    """
    if not filename:
        return "default"

    cfg = get_config()
    if not cfg.misc.source_regexes:
        return "default"

    if "." in filename:
        filename, _ = splitext(filename)

    for pattern, source in cfg.misc.source_regexes:
        try:
            if re.match(pattern, filename):
                return source
        except re.error as e:
            logging.error(f"Invalid regex pattern '{pattern}': {e}")
            continue

    return "default"


def is_audio_file(file_path):
    """
    Return True iff given path is an audio file
    """
    if os.path.isfile(file_path):
        base, ext = os.path.splitext(file_path)
        if ext is not None and len(ext) > 0 and ext.lower() in AUDIO_EXTS:
            return True

    return False


# =============================================================================
# Spectrogram Functions
# =============================================================================


def compress_spectrogram(spec) -> bytes:
    """
    Compress a spectrogram in preparation for inserting into database.

    Args:
    - spec: Uncompressed spectrogram

    Returns:
        Compressed spectrogram
    """
    import numpy as np
    import torch

    if not isinstance(spec, torch.Tensor):
        raise TypeError("spec must be a torch.Tensor")

    if spec.numel() == 0:
        raise ValueError("spec cannot be empty")

    try:
        bytes_spec = spec.cpu().numpy() * 255
        # Fix: Add bounds checking
        bytes_spec = np.clip(bytes_spec, 0, 255)
        np_bytes = bytes_spec.astype(np.uint8)
        bytes_data = np_bytes.tobytes()
        compressed = zlib.compress(bytes_data)
        return compressed
    except Exception as e:
        raise RuntimeError(f"Failed to compress spectrogram: {e}")


def expand_spectrogram(spec: bytes):
    """
    Decompress a spectrogram, then convert from bytes to floats and reshape it.

    Args:
    - spec: Compressed spectrogram

    Returns:
        Uncompressed spectrogram
    """
    import numpy as np

    if not isinstance(spec, bytes):
        raise TypeError("spec must be bytes")

    try:
        cfg = get_config()
        bytes_data = zlib.decompress(spec)
        spec_array = np.frombuffer(bytes_data, dtype=np.uint8) / 255
        spec_array = spec_array.astype(np.float32)

        # Fix: Add validation for expected shape
        expected_size = cfg.audio.spec_height * cfg.audio.spec_width
        if spec_array.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} elements, got {spec_array.size}"
            )

        spec_array = spec_array.reshape(1, cfg.audio.spec_height, cfg.audio.spec_width)
        return spec_array
    except zlib.error as e:
        raise RuntimeError(f"Failed to decompress spectrogram: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to expand spectrogram: {e}")


# =============================================================================
# Label Processing Functions
# =============================================================================


def select_label_regex(line: str) -> Tuple[Optional[re.Pattern], bool]:
    """
    Given a label line, choose a suitable regular expression to parse it.
    """
    if not line:
        return None, False

    label_regex = None
    is_birdnet = False

    try:
        pattern = re.compile("(\\S+)\\t(\\S+)\\t([\\S ]+);(\\S+)")
        if pattern.match(line):
            # Britekit labels
            label_regex = pattern
        else:
            pattern = re.compile("(\\S+)\\t(\\S+)\\t[\\S ]+\\,\\s([\\S ]+)\\t(\\S+)")
            if pattern.match(line):
                # BirdNET labels with --rtype audacity
                label_regex = pattern
                is_birdnet = True
    except re.error as e:
        logging.error(f"Invalid regex pattern: {e}")
        return None, False

    return label_regex, is_birdnet


def labels_to_list(input_path: str) -> List[SimpleNamespace]:
    """
    Given a directory containing Audacity label files generated by BriteKit, BirdNET or Perch,
    return a list of labels.
    """
    if not input_path or not os.path.exists(input_path):
        return []

    label_regex = None
    label_list = []

    try:
        label_paths = glob.glob(os.path.join(input_path, "*.txt"))
        for label_path in label_paths:
            # get the file_prefix, i.e. stem minus suffix, which should match the recording name stem
            if label_path.endswith(".BirdNET.results.txt"):
                name = Path(label_path).name
                file_prefix = name[: -len(".BirdNET.results.txt")]
            elif "_" in label_path:
                name = Path(label_path).name
                file_prefix = name[: name.rfind("_")]
            else:
                continue  # ignore this one

            lines = get_file_lines(label_path)
            for line in lines:
                if label_regex is None:
                    label_regex, is_birdnet = select_label_regex(line)
                    if label_regex is None:
                        logging.error(f"Unknown label format in {label_path}: {line}")
                        continue

                try:
                    if is_birdnet:
                        result = re.split(label_regex, line)
                        if len(result) != 6:
                            continue

                        start_time = float(result[1])
                        end_time = float(result[2])
                        name = result[3]
                        score = float(result[4])
                    else:
                        # this is faster than regex for parsing BriteKit-format labels
                        tokens = line.split("\t")
                        if len(tokens) != 3:
                            continue

                        start_time = float(tokens[0])
                        end_time = float(tokens[1])
                        tokens2 = tokens[2].split(";")
                        if len(tokens2) != 2:
                            continue
                        name = tokens2[0]
                        score = float(tokens2[1])

                    label_list.append(
                        SimpleNamespace(
                            recording=file_prefix,
                            name=name,
                            start_time=start_time,
                            end_time=end_time,
                            score=score,
                        )
                    )
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing line '{line}' in {label_path}: {e}")
                    continue
    except Exception as e:
        logging.error(f"Error processing labels in {input_path}: {e}")
        return []

    return label_list


def labels_to_dataframe(input_path: str):
    """
    Given a directory containing Audacity label files generated by BriteKit, BirdNET or Perch,
    return a Pandas dataframe representing the labels.
    """
    import pandas as pd

    labels = labels_to_list(input_path)

    # Fix: Handle empty labels case
    if not labels:
        return pd.DataFrame(
            columns=["recording", "name", "start_time", "end_time", "score"]
        )

    recordings = []
    names = []
    start_times = []
    end_times = []
    scores = []

    for label in labels:
        recordings.append(label.recording)
        names.append(label.name)
        start_times.append(label.start_time)
        end_times.append(label.end_time)
        scores.append(label.score)

    df = pd.DataFrame()
    df["recording"] = recordings
    df["name"] = names
    df["start_time"] = start_times
    df["end_time"] = end_times
    df["score"] = scores

    return df


def inference_output_to_dataframe(input_path: str):
    """
    Given a directory containing either a CSV file or Audacity label files,
    return a Pandas dataframe representing the labels.
    """
    import pandas as pd

    if not input_path or not os.path.exists(input_path):
        raise InputError(f"Input path does not exist: {input_path}")

    try:
        csv_files = glob.glob(os.path.join(input_path, "*.csv"))
        if len(csv_files) > 1:
            for csv_file in csv_files:
                if csv_file.endswith("_labels.csv"):
                    df = pd.read_csv(csv_file)
                    return df

            raise InputError(
                f"Error: multiple CSV files found in directory {input_path}, but none end with '_labels.csv'."
            )
        elif len(csv_files) == 1:
            try:
                df = pd.read_csv(csv_files[0])
                return df
            except Exception as e:
                raise InputError(f"Error reading CSV file {csv_files[0]}: {e}")
        else:
            # Use the Audacity label files
            df = labels_to_dataframe(input_path)
            return df
    except Exception as e:
        if isinstance(e, InputError):
            raise
        raise InputError(f"Error processing inference output in {input_path}: {e}")
