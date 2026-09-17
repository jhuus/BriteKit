import pickle
import struct
import zlib

import numpy as np
import pytest

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import get_config, set_base_config
from britekit.core.dataset import SpectrogramDataset
from britekit.core.audio_util import convert_to_db
from britekit.core.pickler import TrainingPickler
from britekit.core.reextractor import Reextractor
from britekit.core.util import compress_spectrogram, expand_spectrogram
from britekit.training_db.extractor import Extractor
from britekit.training_db.training_db import TrainingDatabase


@pytest.fixture
def cfg():
    config = BaseConfig()
    config.audio.spec_height = 2
    config.audio.spec_width = 4
    config.train.augment = False
    set_base_config(config)
    yield config
    set_base_config(BaseConfig())


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("reader_bits", [8, 16])
def test_precision_roundtrip_is_independent_of_reader_setting(cfg, bits, reader_bits):
    values = np.array([[0, 0.0001, 0.001, 0.1], [0.2, 0.5, 0.9, 1]], dtype=np.float32)
    encoded = compress_spectrogram(values, bits=bits)
    cfg.audio.spec_bits = reader_bits
    actual = expand_spectrogram(encoded)
    assert actual.dtype == np.float32
    assert actual.shape == (1, 2, 4)
    np.testing.assert_allclose(actual[0], values, atol=1 / (2**bits - 1), rtol=0)
    assert (actual[0, 0, 1] > 0) == (bits == 16)


def test_default_writer_is_byte_compatible_with_legacy_format(cfg):
    assert cfg.audio.spec_bits == 8
    values = np.linspace(-0.2, 1.2, 8, dtype=np.float32).reshape(2, 4)
    legacy = zlib.compress(np.clip(values * 255, 0, 255).astype(np.uint8).tobytes())
    assert compress_spectrogram(values) == legacy
    cfg.audio.spec_bits = 16
    np.testing.assert_array_equal(
        expand_spectrogram(legacy)[0],
        np.clip(values * 255, 0, 255).astype(np.uint8).astype(np.float32) / 255,
    )


def test_16_bit_headered_wire_format(cfg):
    bits = 16
    values = np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.float32)
    encoded = compress_spectrogram(values, bits=bits)
    assert encoded[:14] == struct.pack("<4sBBII", b"BKSP", 1, bits, 2, 4)
    assert zlib.decompress(encoded[14:]) == struct.pack("<HH", 0, 65535) * 4


@pytest.mark.parametrize(
    "blob,match",
    [
        (b"BKSP", "Truncated"),
        (struct.pack("<4sBBII", b"BKSP", 2, 16, 2, 4), "Unsupported"),
        (struct.pack("<4sBBII", b"BKSP", 1, 7, 2, 4), "Unsupported"),
        (struct.pack("<4sBBII", b"BKSP", 1, 9, 2, 4), "Unsupported"),
        (struct.pack("<4sBBII", b"BKSP", 1, 10, 2, 4), "Unsupported"),
        (struct.pack("<4sBBII", b"BKSP", 1, 12, 2, 4), "Unsupported"),
        (struct.pack("<4sBBII", b"BKSP", 1, 14, 2, 4), "Unsupported"),
        (struct.pack("<4sBBII", b"BKSP", 1, 16, 4, 2), "shape"),
        (
            struct.pack("<4sBBII", b"BKSP", 1, 16, 2, 4) + zlib.compress(b"\0\0"),
            "Expected",
        ),
    ],
)
def test_bad_headers_and_payloads_fail_clearly(cfg, blob, match):
    with pytest.raises(RuntimeError, match=match):
        expand_spectrogram(blob)


@pytest.mark.parametrize("bits", [7, 9, 10, 12, 14])
def test_invalid_precision_fails_at_config_load(cfg, bits):
    cfg.audio.spec_bits = bits
    with pytest.raises(ValueError, match="spec_bits"):
        get_config()
    with pytest.raises(ValueError, match="8 or 16"):
        compress_spectrogram(np.zeros((2, 4)), bits=bits)


@pytest.mark.parametrize("rewrite_bits", [8, 16])
def test_extraction_pickle_mixed_precision_training_and_reextraction(
    cfg, tmp_path, monkeypatch, rewrite_bits
):
    values = np.array([[0, 0.0001, 0.001, 0.1], [0.2, 0.5, 0.9, 1]], dtype=np.float32)
    calls = []

    class FakeAudio:
        def load(self, path):
            pass

        def seconds(self):
            return 10

        def get_spectrograms(self, offsets, **kwargs):
            calls.append(kwargs)
            assert kwargs["convert_to_db"] is False
            return np.stack([values for _ in offsets]), None

    monkeypatch.setattr("britekit.core.audio.Audio", FakeAudio)
    db_path = str(tmp_path / "training.db")
    recording = tmp_path / "recording.wav"
    recording.touch()
    cfg.audio.convert_to_db = True
    with TrainingDatabase(db_path) as db:
        extractor = Extractor(db, "Bird", "BIRD")
        extractor.insert_spectrograms(str(recording), [0])
        cfg.audio.spec_bits = 16
        extractor.insert_spectrograms(str(recording), [8])
        blobs = [row.value for row in db.get_specvalue()]
        assert not blobs[0].startswith(b"BKSP")
        assert len(blobs) == 2
        assert struct.unpack("<4sBBII", blobs[1][:14])[2] == 16

    output = str(tmp_path / "training.pkl")
    writer = TrainingPickler(db_path, output)
    try:
        writer.pickle(quiet=True)
    finally:
        writer.db.close()
    with open(output, "rb") as stream:
        data = pickle.load(stream)
    assert set(data["spec_values"]) == set(blobs)
    dataset = SpectrogramDataset(data["spec_values"], data["spec_class_indexes"], 1)
    for index, blob in enumerate(data["spec_values"]):
        np.testing.assert_allclose(
            dataset[index]["input"], convert_to_db(expand_spectrogram(blob))
        )

    # Re-extraction rewrites precision; pickling preserves mixed-precision blobs.
    cfg.audio.spec_bits = rewrite_bits
    Reextractor(db_path=db_path).run(quiet=True)
    with TrainingDatabase(db_path) as db:
        assert all(
            row.value == compress_spectrogram(values, bits=rewrite_bits)
            for row in db.get_specvalue()
        )
    assert len(calls) >= 3


def test_storage_precision_does_not_quantize_audio_inference(cfg):
    from britekit.core.audio import Audio

    audio = Audio(device="cpu", cfg=cfg)
    audio.signal = np.zeros(5 * cfg.audio.sampling_rate, dtype=np.float32)
    values = np.array([[0, 0.0001, 0.001, 0.1], [0.2, 0.5, 0.9, 1]], dtype=np.float32)
    audio._get_spectrograms_sliced = lambda *args: [values.copy()]
    first, _ = audio.get_spectrograms([0], convert_to_db=True)
    for bits in (8, 16):
        cfg.audio.spec_bits = bits
        second, _ = audio.get_spectrograms([0], convert_to_db=True)
        np.testing.assert_array_equal(first, second)


def test_16_bit_writer_handles_float16_input_without_overflow(cfg):
    values = np.linspace(0, 1, 8, dtype=np.float16).reshape(2, 4)
    decoded = expand_spectrogram(compress_spectrogram(values, bits=16))
    np.testing.assert_allclose(decoded[0], values, atol=1 / 65535, rtol=0)


@pytest.mark.parametrize("stored_bits", [None, 8, 16])
def test_checkpoint_precision_metadata_is_backward_compatible(cfg, stored_bits):
    from types import SimpleNamespace
    from britekit.core.util import cfg_to_pure
    from britekit.models.base_model import BaseModel

    metadata = cfg_to_pure(cfg)
    if stored_bits is None:
        del metadata["audio"]["spec_bits"]
    else:
        metadata["audio"]["spec_bits"] = stored_bits
    model = SimpleNamespace(training_cfg=metadata)
    resolved = BaseConfig()
    BaseModel.apply_training_config(model, resolved)
    assert resolved.audio.spec_bits == (8 if stored_bits is None else stored_bits)
