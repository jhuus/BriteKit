"""Bulk decoding must preserve stream choice, errors, and damaged-file prefixes."""

import builtins
from contextlib import contextmanager
from types import SimpleNamespace

import av
import numpy as np
import pytest

from britekit.core import audio


@pytest.fixture(autouse=True)
def isolate_decoder_cache():
    factory = audio._get_torchcodec_decoder
    factory.cache_clear()
    yield
    factory.cache_clear()


def test_native_uses_first_audio_stream_and_returns_contiguous_pcm(monkeypatch):
    expected = np.array([[0.25, 0.5], [-0.25, -0.5]], dtype=np.float32)
    calls = []

    @contextmanager
    def open_container(path, **kwargs):
        assert kwargs == {"metadata_errors": "replace"}
        yield SimpleNamespace(
            streams=SimpleNamespace(
                audio=[SimpleNamespace(index=3), SimpleNamespace(index=5)]
            )
        )

    class Decoder:
        def __init__(self, path, *, stream_index):
            calls.append((path, stream_index))

        def get_all_samples(self):
            return SimpleNamespace(
                data=SimpleNamespace(numpy=lambda: expected[:, ::-1]),
                sample_rate=32000,
            )

    monkeypatch.setattr(av, "open", open_container)
    monkeypatch.setattr(audio, "_get_torchcodec_decoder", lambda: Decoder)
    signal, rate = audio._decode_audio("multiple-streams.mka")

    assert calls == [("multiple-streams.mka", 3)]
    assert rate == 32000
    assert signal.dtype == np.float32
    assert signal.flags.c_contiguous
    np.testing.assert_array_equal(signal, expected[:, ::-1])


@pytest.mark.parametrize("error_type", [ImportError, OSError, RuntimeError])
def test_unavailable_native_runtime_is_cached(monkeypatch, error_type):
    original_import = builtins.__import__
    attempts = []

    def import_module(name, *args, **kwargs):
        if name == "torchcodec.decoders":
            attempts.append(name)
            raise error_type("Incompatible native runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    expected = (np.ones((1, 10), dtype=np.float32), 32000)
    monkeypatch.setattr(audio, "_decode_audio_pyav", lambda path: expected)

    assert audio._decode_audio("first.mp3") is expected
    assert audio._decode_audio("second.mp3") is expected
    assert attempts == ["torchcodec.decoders"]


def test_failed_native_decode_preserves_pyav_error(monkeypatch, tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    original = ValueError("Original PyAV failure")

    def fail(path):
        raise original

    monkeypatch.setattr(audio, "_decode_audio_pyav", fail)
    with pytest.raises(ValueError) as caught:
        audio._decode_audio(str(path))
    assert caught.value is original


@pytest.fixture(scope="module")
def encoded_mp3(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio-decoder") / "recording.mp3"
    rate = 32000
    signal = (0.2 * np.sin(2 * np.pi * 440 * np.arange(rate * 4) / rate)).astype(
        np.float32
    )
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libmp3lame", rate=rate)
        stream.layout = "mono"
        stream.bit_rate = 192000
        frame = av.AudioFrame.from_ndarray(
            signal[None, :], format="fltp", layout="mono"
        )
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def test_clean_mp3_uses_native_without_recovery(monkeypatch, encoded_mp3):
    if audio._get_torchcodec_decoder() is None:
        pytest.skip("Compatible TorchCodec/FFmpeg runtime unavailable")
    expected, rate = audio._decode_audio_pyav(str(encoded_mp3))

    def unexpected_recovery(path):
        pytest.fail("Clean MP3 should use TorchCodec")

    monkeypatch.setattr(audio, "_decode_audio_pyav", unexpected_recovery)
    actual, actual_rate = audio._decode_audio(str(encoded_mp3))
    assert actual_rate == rate
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("location", ["middle", "tail"])
@pytest.mark.parametrize("damage", ["side_info", "header"])
def test_real_mp3_damage_keeps_only_prefix(
    tmp_path, encoded_mp3, monkeypatch, location, damage
):
    if audio._get_torchcodec_decoder() is None:
        pytest.skip("Compatible TorchCodec/FFmpeg runtime unavailable")
    with av.open(str(encoded_mp3)) as container:
        packets = [
            (packet.pos, packet.size)
            for packet in container.demux(container.streams.audio[0])
            if packet.size
        ]
    index = len(packets) // 2 if location == "middle" else len(packets) - 2
    position, size = packets[index]
    data = bytearray(encoded_mp3.read_bytes())
    if damage == "side_info":
        data[position + 4 : position + 36] = b"\xff" * 32
    else:
        data[position : position + 4] = b"\0" * 4
    path = tmp_path / "damaged.mp3"
    path.write_bytes(data)
    original_decode = audio._decode_audio_pyav
    expected, rate = original_decode(str(path))
    clean, _ = original_decode(str(encoded_mp3))
    assert 0 < expected.shape[1] < clean.shape[1]
    assert expected.tobytes() == clean[:, : expected.shape[1]].tobytes()
    calls = []

    def recover(path):
        calls.append(path)
        return original_decode(path)

    monkeypatch.setattr(audio, "_decode_audio_pyav", recover)
    actual, actual_rate = audio._decode_audio(str(path))
    assert calls == [str(path)]
    assert actual_rate == rate
    assert actual.tobytes() == expected.tobytes()
