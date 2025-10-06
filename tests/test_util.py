from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from britekit.core.util import (
    format_elapsed_time,
    get_range,
    cfg_to_pure,
    cli_help_from_doc,
    get_audio_files,
    get_file_lines,
    get_source_name,
    compress_spectrogram,
    expand_spectrogram,
    select_label_regex,
    labels_to_list,
    labels_to_dataframe,
    inference_output_to_dataframe,
    AUDIO_EXTS,
)
from britekit.core.exceptions import InputError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_spectrogram():
    """Create a sample spectrogram tensor for testing."""
    return torch.randn(64, 128, dtype=torch.float32)


@pytest.fixture
def sample_audio_files(temp_dir):
    """Create sample audio files for testing."""
    audio_files = []
    for ext in [".mp3", ".wav", ".flac", ".txt"]:  # .txt is not audio
        file_path = os.path.join(temp_dir, f"test_file{ext}")
        with open(file_path, "w") as f:
            f.write("test content")
        if ext in AUDIO_EXTS:
            audio_files.append(file_path)
    return temp_dir, audio_files


@pytest.fixture
def sample_label_files(temp_dir):
    """Create sample label files for testing."""
    # Create BriteKit format label file
    britekit_file = os.path.join(temp_dir, "recording_1.txt")
    with open(britekit_file, "w") as f:
        f.write("0.0\t1.0\tCommonYellowthroat;0.95\n")
        f.write("1.5\t2.5\tRuffedGrouse;0.87\n")

    return temp_dir


@pytest.fixture
def sample_birdnet_files(temp_dir):
    """Create sample BirdNET format label files for testing."""
    # Create BirdNET format label file
    birdnet_file = os.path.join(temp_dir, "recording_2.BirdNET.results.txt")
    with open(birdnet_file, "w") as f:
        f.write("0.0\t1.0\tGeothlypis trichas, Common Yellowthroat\t0.95\n")
        f.write("1.5\t2.5\tBonasa umbellus, Ruffed Grouse\t0.87\n")

    return temp_dir


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    csv_file = os.path.join(temp_dir, "test_labels.csv")
    df = pd.DataFrame(
        {
            "recording": ["test1", "test2"],
            "name": ["CommonYellowthroat", "RuffedGrouse"],
            "start_time": [0.0, 1.5],
            "end_time": [1.0, 2.5],
            "score": [0.95, 0.87],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


# =============================================================================
# Test Classes for cfg_to_pure
# =============================================================================


@dataclass
class TestConfig:
    """Test dataclass for cfg_to_pure testing."""

    name: str
    value: int
    nested: dict


class TestEnum(Enum):
    """Test enum for cfg_to_pure testing."""

    VALUE1 = "value1"
    VALUE2 = "value2"


# =============================================================================
# Core Utility Functions Tests
# =============================================================================


class TestFormatElapsedTime:
    """Test format_elapsed_time function."""

    def test_format_elapsed_time_seconds_only(self):
        """Test formatting time less than 1 hour."""
        result = format_elapsed_time(0, 125)
        assert result == "02:05"

    def test_format_elapsed_time_with_hours(self):
        """Test formatting time with hours."""
        result = format_elapsed_time(0, 7325)  # 2 hours, 2 minutes, 5 seconds
        assert result == "02:02:05"

    def test_format_elapsed_time_zero(self):
        """Test formatting zero elapsed time."""
        result = format_elapsed_time(100, 100)
        assert result == "00:00"

    def test_format_elapsed_time_invalid_input(self):
        """Test formatting with invalid input."""
        with pytest.raises(
            ValueError, match="end_time must be greater than or equal to start_time"
        ):
            format_elapsed_time(100, 50)


class TestGetRange:
    """Test get_range function."""

    def test_get_range_basic(self):
        """Test basic range generation."""
        result = get_range(0, 10, 2)
        expected = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        assert result == expected

    def test_get_range_single_value(self):
        """Test range with zero increment."""
        result = get_range(5, 5, 0)
        assert result == [5.0]

    def test_get_range_negative_increment(self):
        """Test range with negative increment."""
        with pytest.raises(ValueError, match="increment must be positive"):
            get_range(0, 10, -1)

    def test_get_range_invalid_bounds(self):
        """Test range with invalid bounds."""
        with pytest.raises(
            ValueError, match="max_val must be greater than or equal to min_val"
        ):
            get_range(10, 5, 1)

    def test_get_range_floating_point(self):
        """Test range with floating point values."""
        result = get_range(0.0, 1.0, 0.25)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert result == expected


# =============================================================================
# Configuration Functions Tests
# =============================================================================


class TestCfgToPure:
    """Test cfg_to_pure function."""

    def test_cfg_to_pure_dataclass(self):
        """Test converting dataclass to pure format."""
        config = TestConfig("test", 42, {"nested": "value"})
        result = cfg_to_pure(config)
        expected = {"name": "test", "value": 42, "nested": {"nested": "value"}}
        assert result == expected

    def test_cfg_to_pure_enum(self):
        """Test converting enum to pure format."""
        result = cfg_to_pure(TestEnum.VALUE1)
        assert result == "value1"

    def test_cfg_to_pure_path(self):
        """Test converting Path to pure format."""
        path = Path("/test/path")
        result = cfg_to_pure(path)
        assert result == "/test/path"

    def test_cfg_to_pure_numpy_array(self):
        """Test converting numpy array to pure format."""
        arr = np.array([1, 2, 3])
        result = cfg_to_pure(arr)
        assert result == [1, 2, 3]

    def test_cfg_to_pure_numpy_scalar(self):
        """Test converting numpy scalar to pure format."""
        scalar = np.int64(42)
        result = cfg_to_pure(scalar)
        assert result == 42

    def test_cfg_to_pure_torch_tensor(self):
        """Test converting torch tensor to pure format."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = cfg_to_pure(tensor)
        expected = {"tensor": {"shape": [2, 2], "dtype": "torch.int64"}}
        assert result == expected

    def test_cfg_to_pure_torch_dtype(self):
        """Test converting torch dtype to pure format."""
        result = cfg_to_pure(torch.float32)
        assert result == "torch.float32"

    def test_cfg_to_pure_torch_device(self):
        """Test converting torch device to pure format."""
        result = cfg_to_pure(torch.device("cpu"))
        assert result == "cpu"

    def test_cfg_to_pure_torch_size(self):
        """Test converting torch Size to pure format."""
        size = torch.Size([1, 2, 3])
        result = cfg_to_pure(size)
        assert result == [1, 2, 3]

    def test_cfg_to_pure_dict(self):
        """Test converting dict to pure format."""
        data = {"key": "value", "number": 42}
        result = cfg_to_pure(data)
        assert result == {"key": "value", "number": 42}

    def test_cfg_to_pure_list(self):
        """Test converting list to pure format."""
        data = [1, "string", {"nested": "value"}]
        result = cfg_to_pure(data)
        assert result == [1, "string", {"nested": "value"}]

    def test_cfg_to_pure_tuple(self):
        """Test converting tuple to pure format."""
        data = (1, 2, 3)
        result = cfg_to_pure(data)
        assert result == [1, 2, 3]

    def test_cfg_to_pure_set(self):
        """Test converting set to pure format."""
        data = {1, 2, 3}
        result = cfg_to_pure(data)
        assert sorted(result) == [1, 2, 3]

    def test_cfg_to_pure_callable(self):
        """Test converting callable to pure format."""

        def test_func():
            pass

        result = cfg_to_pure(test_func)
        assert "test_func" in result

    def test_cfg_to_pure_class(self):
        """Test converting class to pure format."""
        result = cfg_to_pure(TestConfig)
        assert "TestConfig" in result

    def test_cfg_to_pure_primitive(self):
        """Test converting primitive types."""
        assert cfg_to_pure("string") == "string"
        assert cfg_to_pure(42) == 42
        assert cfg_to_pure(3.14) == 3.14
        assert cfg_to_pure(True) is True
        assert cfg_to_pure(None) is None

    def test_cfg_to_pure_recursion_limit(self):
        """Test recursion limit handling."""
        # Create a circular reference
        data = {}
        data["self"] = data

        result = cfg_to_pure(data)
        # The function should handle circular references gracefully
        # The exact behavior depends on the implementation, but it shouldn't crash
        assert isinstance(result, dict)
        # The circular reference might be preserved as-is or converted
        # Let's just check that we get a valid result without infinite recursion
        assert "self" in result


# =============================================================================
# Documentation/CLI Functions Tests
# =============================================================================


class TestCliHelpFromDoc:
    """Test cli_help_from_doc function."""

    def test_cli_help_from_doc_none(self):
        """Test with None input."""
        result = cli_help_from_doc(None)
        assert result is None

    def test_cli_help_from_doc_no_args_section(self):
        """Test docstring without args section."""
        doc = "This is a test docstring.\n\nIt has multiple lines."
        result = cli_help_from_doc(doc)
        assert result == "This is a test docstring.\n\nIt has multiple lines."

    def test_cli_help_from_doc_with_args_section(self):
        """Test docstring with args section."""
        doc = """This is a test docstring.

Args:
    param1: Description of param1
    param2: Description of param2

Returns:
    Description of return value"""
        result = cli_help_from_doc(doc)
        assert result == "This is a test docstring."

    def test_cli_help_from_doc_with_arguments_section(self):
        """Test docstring with 'Arguments' section."""
        doc = """This is a test docstring.

Arguments:
    param1: Description of param1"""
        result = cli_help_from_doc(doc)
        assert result == "This is a test docstring."

    def test_cli_help_from_doc_with_parameters_section(self):
        """Test docstring with 'Parameters' section."""
        doc = """This is a test docstring.

Parameters:
    param1: Description of param1"""
        result = cli_help_from_doc(doc)
        assert result == "This is a test docstring."


# =============================================================================
# File System Functions Tests
# =============================================================================


class TestGetAudioFiles:
    """Test get_audio_files function."""

    def test_get_audio_files_empty_path(self):
        """Test with empty path."""
        result = get_audio_files("")
        assert result == []

    def test_get_audio_files_nonexistent_path(self):
        """Test with nonexistent path."""
        result = get_audio_files("/nonexistent/path")
        assert result == []

    def test_get_audio_files_with_files(self, sample_audio_files):
        """Test with actual audio files."""
        temp_dir, expected_files = sample_audio_files
        result = get_audio_files(temp_dir, short_names=False)

        # Check that we get the expected number of audio files
        assert len(result) == len(expected_files)

        # Check that all returned files are audio files
        for file_path in result:
            _, ext = os.path.splitext(file_path)
            assert ext.lower() in AUDIO_EXTS

    def test_get_audio_files_short_names(self, sample_audio_files):
        """Test with short_names=True."""
        temp_dir, _ = sample_audio_files
        result = get_audio_files(temp_dir, short_names=True)

        # Check that we get filenames without full paths
        for file_name in result:
            assert not os.path.isabs(file_name)
            assert os.path.basename(file_name) == file_name

    def test_get_audio_files_permission_error(self, temp_dir):
        """Test handling of permission errors."""
        # Create a directory that we can't read
        restricted_dir = os.path.join(temp_dir, "restricted")
        os.makedirs(restricted_dir, mode=0o000)

        try:
            result = get_audio_files(restricted_dir)
            assert result == []
        finally:
            # Clean up
            os.chmod(restricted_dir, 0o755)


class TestGetFileLines:
    """Test get_file_lines function."""

    def test_get_file_lines_empty_path(self):
        """Test with empty path."""
        result = get_file_lines("")
        assert result == []

    def test_get_file_lines_nonexistent_file(self):
        """Test with nonexistent file."""
        result = get_file_lines("/nonexistent/file.txt")
        assert result == []

    def test_get_file_lines_basic_content(self, temp_dir):
        """Test reading basic file content."""
        file_path = os.path.join(temp_dir, "test.txt")
        content = "line1\nline2\nline3\n"
        with open(file_path, "w") as f:
            f.write(content)

        result = get_file_lines(file_path)
        assert result == ["line1", "line2", "line3"]

    def test_get_file_lines_with_comments(self, temp_dir):
        """Test reading file with comments."""
        file_path = os.path.join(temp_dir, "test.txt")
        content = "line1\n# comment\nline2\n  # indented comment\nline3\n"
        with open(file_path, "w") as f:
            f.write(content)

        result = get_file_lines(file_path)
        assert result == ["line1", "line2", "line3"]

    def test_get_file_lines_with_whitespace(self, temp_dir):
        """Test reading file with whitespace."""
        file_path = os.path.join(temp_dir, "test.txt")
        content = "  line1  \n\nline2\n  \nline3\n"
        with open(file_path, "w") as f:
            f.write(content)

        result = get_file_lines(file_path)
        assert result == ["line1", "line2", "line3"]

    def test_get_file_lines_encoding_error(self, temp_dir):
        """Test handling of encoding errors."""
        file_path = os.path.join(temp_dir, "test.txt")
        # Write binary data that's not valid UTF-8
        with open(file_path, "wb") as f:
            f.write(b"\xff\xfe\x00\x00")  # Invalid UTF-8

        result = get_file_lines(file_path)
        assert result == []


class TestGetSourceName:
    """Test get_source_name function."""

    @patch("britekit.core.util.get_config")
    def test_get_source_name_empty_filename(self, mock_get_config):
        """Test with empty filename."""
        mock_cfg = MagicMock()
        mock_cfg.misc.source_regexes = []
        mock_get_config.return_value = (mock_cfg, None)

        result = get_source_name("")
        assert result == "default"

    @patch("britekit.core.util.get_config")
    def test_get_source_name_no_regexes(self, mock_get_config):
        """Test when no regexes are configured."""
        mock_cfg = MagicMock()
        mock_cfg.misc.source_regexes = None
        mock_get_config.return_value = (mock_cfg, None)

        result = get_source_name("test.mp3")
        assert result == "default"

    @patch("britekit.core.util.get_config")
    def test_get_source_name_matching_regex(self, mock_get_config):
        """Test with matching regex."""
        mock_cfg = MagicMock()
        mock_cfg.misc.source_regexes = [("test.*", "test_source")]
        mock_get_config.return_value = (mock_cfg, None)

        result = get_source_name("test_file.mp3")
        assert result == "test_source"

    @patch("britekit.core.util.get_config")
    def test_get_source_name_no_match(self, mock_get_config):
        """Test when no regex matches."""
        mock_cfg = MagicMock()
        mock_cfg.misc.source_regexes = [("test.*", "test_source")]
        mock_get_config.return_value = (mock_cfg, None)

        result = get_source_name("other_file.mp3")
        assert result == "default"

    @patch("britekit.core.util.get_config")
    def test_get_source_name_invalid_regex(self, mock_get_config):
        """Test with invalid regex pattern."""
        mock_cfg = MagicMock()
        mock_cfg.misc.source_regexes = [("[invalid", "test_source")]
        mock_get_config.return_value = (mock_cfg, None)

        result = get_source_name("test_file.mp3")
        assert result == "default"

    @patch("britekit.core.util.get_config")
    def test_get_source_name_with_extension(self, mock_get_config):
        """Test filename with extension."""
        mock_cfg = MagicMock()
        mock_cfg.misc.source_regexes = [("test.*", "test_source")]
        mock_get_config.return_value = (mock_cfg, None)

        result = get_source_name("test_file.mp3")
        assert result == "test_source"


# =============================================================================
# Spectrogram Functions Tests
# =============================================================================


class TestCompressSpectrogram:
    """Test compress_spectrogram function."""

    def test_compress_spectrogram_valid_tensor(self, sample_spectrogram):
        """Test compressing a valid tensor."""
        result = compress_spectrogram(sample_spectrogram)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_compress_spectrogram_invalid_type(self):
        """Test with invalid input type."""
        with pytest.raises(TypeError, match="spec must be a torch.Tensor"):
            compress_spectrogram("not a tensor")

    def test_compress_spectrogram_empty_tensor(self):
        """Test with empty tensor."""
        empty_tensor = torch.tensor([])
        with pytest.raises(ValueError, match="spec cannot be empty"):
            compress_spectrogram(empty_tensor)

    def test_compress_spectrogram_bounds_checking(self):
        """Test bounds checking for values outside [0, 1]."""
        # Create tensor with values outside [0, 1]
        tensor = torch.tensor([[-1.0, 0.5, 2.0]])
        result = compress_spectrogram(tensor)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestExpandSpectrogram:
    """Test expand_spectrogram function."""

    @patch("britekit.core.util.get_config")
    def test_expand_spectrogram_valid_bytes(self, mock_get_config, sample_spectrogram):
        """Test expanding valid compressed data."""
        # Mock config to match our test spectrogram dimensions
        mock_cfg = MagicMock()
        mock_cfg.audio.spec_height = 64
        mock_cfg.audio.spec_width = 128
        mock_get_config.return_value = (mock_cfg, None)

        compressed = compress_spectrogram(sample_spectrogram)
        result = expand_spectrogram(compressed)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (64, 128, 1)  # Based on sample_spectrogram shape

    def test_expand_spectrogram_invalid_type(self):
        """Test with invalid input type."""
        with pytest.raises(TypeError, match="spec must be bytes"):
            expand_spectrogram("not bytes")

    def test_expand_spectrogram_invalid_compression(self):
        """Test with invalid compressed data."""
        with pytest.raises(RuntimeError, match="Failed to decompress spectrogram"):
            expand_spectrogram(b"invalid compressed data")

    @patch("britekit.core.util.get_config")
    def test_expand_spectrogram_wrong_size(self, mock_get_config):
        """Test with wrong size data."""
        # Create a spectrogram with specific dimensions
        test_spectrogram = torch.randn(32, 64, dtype=torch.float32)

        # Mock config to expect different dimensions than what we compressed
        mock_cfg = MagicMock()
        mock_cfg.audio.spec_height = 64  # Different from test_spectrogram height (32)
        mock_cfg.audio.spec_width = 128  # Different from test_spectrogram width (64)
        mock_get_config.return_value = (mock_cfg, None)

        # Compress with one size, try to expand with different expected size
        compressed = compress_spectrogram(test_spectrogram)

        with pytest.raises(
            RuntimeError,
            match="Failed to expand spectrogram: Expected 8192 elements, got 2048",
        ):
            expand_spectrogram(compressed)


# =============================================================================
# Label Processing Functions Tests
# =============================================================================


class TestSelectLabelRegex:
    """Test select_label_regex function."""

    def test_select_label_regex_empty_line(self):
        """Test with empty line."""
        result = select_label_regex("")
        assert result == (None, False)

    def test_select_label_regex_britekit_format(self):
        """Test BriteKit format line."""
        line = "0.0\t1.0\tCommonYellowthroat;0.95"
        result = select_label_regex(line)
        assert result[0] is not None
        assert result[1] is False

    def test_select_label_regex_birdnet_format(self):
        """Test BirdNET format line."""
        line = "0.0\t1.0\tCommon Yellowthroat, Geothlypis trichas\t0.95"
        result = select_label_regex(line)
        assert result[0] is not None
        assert result[1] is True

    def test_select_label_regex_unknown_format(self):
        """Test unknown format line."""
        line = "unknown format line"
        result = select_label_regex(line)
        assert result == (None, False)

    def test_select_label_regex_invalid_regex(self):
        """Test with invalid regex pattern."""
        # This shouldn't happen with the current implementation, but test for robustness
        with patch(
            "britekit.core.util.re.compile", side_effect=re.error("Invalid pattern")
        ):
            result = select_label_regex("0.0\t1.0\tCommonYellowthroat;0.95")
            assert result == (None, False)


class TestLabelsToList:
    """Test labels_to_list function."""

    def test_labels_to_list_empty_path(self):
        """Test with empty path."""
        result = labels_to_list("")
        assert result == []

    def test_labels_to_list_nonexistent_path(self):
        """Test with nonexistent path."""
        result = labels_to_list("/nonexistent/path")
        assert result == []

    def test_labels_to_list_britekit_format(self, sample_label_files):
        """Test parsing BriteKit format labels."""
        result = labels_to_list(sample_label_files)
        assert len(result) == 2  # 2 BriteKit labels

        # Check BriteKit labels (recording - the function strips the _1 suffix)
        assert result[0].recording == "recording"
        assert result[0].name == "CommonYellowthroat"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 1.0
        assert result[0].score == 0.95

        assert result[1].recording == "recording"
        assert result[1].name == "RuffedGrouse"
        assert result[1].start_time == 1.5
        assert result[1].end_time == 2.5
        assert result[1].score == 0.87

    def test_labels_to_list_birdnet_format(self, sample_birdnet_files):
        """Test parsing BirdNET format labels."""
        result = labels_to_list(sample_birdnet_files)
        assert len(result) == 2  # 2 BirdNET labels

        # Check BirdNET labels
        assert result[0].recording == "recording_2"
        assert result[0].name == "Common Yellowthroat"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 1.0
        assert result[0].score == 0.95

        assert result[1].recording == "recording_2"
        assert result[1].name == "Ruffed Grouse"
        assert result[1].start_time == 1.5
        assert result[1].end_time == 2.5
        assert result[1].score == 0.87

    def test_labels_to_list_invalid_line_format(self, temp_dir):
        """Test handling of invalid line formats."""
        # Create file with invalid format (needs underscore to be processed)
        label_file = os.path.join(temp_dir, "test_invalid.txt")
        with open(label_file, "w") as f:
            f.write("invalid line format\n")
            f.write("0.0\t1.0\tCommonYellowthroat;0.95\n")  # Valid line

        result = labels_to_list(temp_dir)
        assert len(result) == 1  # Only the valid line should be parsed

    def test_labels_to_list_malformed_data(self, temp_dir):
        """Test handling of malformed data."""
        # Create file with malformed data
        label_file = os.path.join(temp_dir, "malformed.txt")
        with open(label_file, "w") as f:
            f.write("0.0\t1.0\tCommonYellowthroat\n")  # Missing score
            f.write("not_a_number\t1.0\tCommonYellowthroat;0.95\n")  # Invalid time

        result = labels_to_list(temp_dir)
        assert result == []  # No valid labels should be parsed


class TestLabelsToDataframe:
    """Test labels_to_dataframe function."""

    def test_labels_to_dataframe_empty_labels(self, temp_dir):
        """Test with empty labels."""
        result = labels_to_dataframe(temp_dir)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "recording",
            "name",
            "start_time",
            "end_time",
            "score",
        ]
        assert len(result) == 0

    def test_labels_to_dataframe_with_labels(self, sample_label_files):
        """Test with actual labels."""
        result = labels_to_dataframe(sample_label_files)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "recording",
            "name",
            "start_time",
            "end_time",
            "score",
        ]
        assert len(result) == 2  # 2 BriteKit labels

        # Check that data is correctly populated
        assert "CommonYellowthroat" in result["name"].values
        assert "RuffedGrouse" in result["name"].values


class TestInferenceOutputToDataframe:
    """Test inference_output_to_dataframe function."""

    def test_inference_output_to_dataframe_empty_path(self):
        """Test with empty path."""
        with pytest.raises(InputError, match="Input path does not exist"):
            inference_output_to_dataframe("")

    def test_inference_output_to_dataframe_nonexistent_path(self):
        """Test with nonexistent path."""
        with pytest.raises(InputError, match="Input path does not exist"):
            inference_output_to_dataframe("/nonexistent/path")

    def test_inference_output_to_dataframe_csv_file(self, sample_csv_file):
        """Test reading CSV file."""
        csv_dir = os.path.dirname(sample_csv_file)
        result = inference_output_to_dataframe(csv_dir)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == [
            "recording",
            "name",
            "start_time",
            "end_time",
            "score",
        ]

    def test_inference_output_to_dataframe_multiple_csv_files(self, temp_dir):
        """Test with multiple CSV files."""
        # Create two CSV files
        csv1 = os.path.join(temp_dir, "file1.csv")
        csv2 = os.path.join(temp_dir, "file2.csv")

        pd.DataFrame({"test": [1]}).to_csv(csv1, index=False)
        pd.DataFrame({"test": [2]}).to_csv(csv2, index=False)

        with pytest.raises(InputError, match="multiple CSV files found"):
            inference_output_to_dataframe(temp_dir)

    def test_inference_output_to_dataframe_label_files(self, sample_label_files):
        """Test reading label files when no CSV is present."""
        result = inference_output_to_dataframe(sample_label_files)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 BriteKit labels
        assert list(result.columns) == [
            "recording",
            "name",
            "start_time",
            "end_time",
            "score",
        ]

    def test_inference_output_to_dataframe_corrupted_csv(self, temp_dir):
        """Test handling of corrupted CSV file."""
        csv_file = os.path.join(temp_dir, "corrupted.csv")
        # Create a file with invalid UTF-8 bytes to make it truly corrupted
        with open(csv_file, "wb") as f:
            f.write(b"invalid,csv,content\n")
            f.write(b"missing,columns\n")
            f.write(b"\xff\xfe\xfd")  # Invalid UTF-8 bytes

        with pytest.raises(InputError, match="Error reading CSV file.*"):
            inference_output_to_dataframe(temp_dir)
