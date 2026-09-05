"""Exercise packet recovery with real PyAV frames and controlled decoder errors."""

from contextlib import contextmanager
from fractions import Fraction
import struct
from types import SimpleNamespace

import av
import numpy as np
import pytest

from britekit.core.audio import Audio, _decode_audio


class Packet:
    def __init__(self, frames=(), error=None, size=1000, duration=1152):
        self.frames = frames
        self.error = error
        self.size = size
        self.duration = duration
        self.time_base = Fraction(1, 44100)

    def decode(self):
        if self.error is not None:
            raise self.error
        return self.frames


def frame_packet():
    frame = av.AudioFrame.from_ndarray(
        np.full((1, 1152), 0.25, dtype=np.float32), format="fltp", layout="mono"
    )
    frame.sample_rate = 44100
    return Packet(frames=[frame])


def bad_packet(**kwargs):
    return Packet(
        error=av.error.InvalidDataError(1094995529, "Header missing"), **kwargs
    )


def install_container(monkeypatch, packets, codec="mp3float"):
    stream = SimpleNamespace(codec_context=SimpleNamespace(name=codec))

    @contextmanager
    def open_container(path, **kwargs):
        yield SimpleNamespace(
            streams=SimpleNamespace(audio=[stream]), demux=lambda stream: iter(packets)
        )

    monkeypatch.setattr(av, "open", open_container)


@pytest.mark.parametrize("codec", ["mp3", "mp3float"])
def test_short_mp3_tail_preserves_pcm(monkeypatch, caplog, codec):
    install_container(
        monkeypatch,
        [frame_packet(), bad_packet(size=304), bad_packet(size=739), Packet(size=0)],
        codec,
    )

    signal, rate = _decode_audio("recording.mpga")

    assert rate == 44100
    np.testing.assert_array_equal(signal, np.full((1, 1152), 0.25, dtype=np.float32))
    assert "recording.mpga" in caplog.text
    assert "Truncated audio" in caplog.text
    assert "0.026 s of decoded audio" in caplog.text


@pytest.mark.parametrize(
    "case",
    [
        "middle",
        "start",
        "no_audio",
        "non_mp3",
        "large",
        "long",
        "cumulative",
        "missing_duration",
        "zero_duration",
        "negative_duration",
        "missing_time_base",
        "negative_time_base",
        "other_error",
        "flush_error",
        "delayed_frame",
        "empty_packets_exceed_budget",
    ],
)
def test_damage_retains_only_usable_prefix(monkeypatch, caplog, case):
    bad = bad_packet()
    packets = [frame_packet(), bad]
    codec = "mp3float"
    if case == "middle":
        packets.append(frame_packet())
    elif case == "start":
        packets.reverse()
    elif case == "no_audio":
        packets = [bad]
    elif case == "non_mp3":
        codec = "aac"
    elif case == "large":
        bad.size = 4097
    elif case == "long":
        bad.duration = 4411
    elif case == "cumulative":
        packets.extend([bad_packet()] * 4)
    elif case == "missing_duration":
        bad.duration = None
    elif case == "zero_duration":
        bad.duration = 0
    elif case == "negative_duration":
        bad.duration = -1
    elif case == "missing_time_base":
        bad.time_base = None
    elif case == "negative_time_base":
        bad.time_base = Fraction(-1, 44100)
    elif case == "other_error":
        bad.error = RuntimeError("decoder failure")
    elif case == "flush_error":
        packets.append(bad_packet(size=0))
    elif case == "delayed_frame":
        delayed = frame_packet()
        delayed.size = 0
        packets.append(delayed)
    elif case == "empty_packets_exceed_budget":
        packets.append(Packet(size=4096))
    # No packet after damage should be decoded, even to flush the decoder.
    packets.append(Packet(error=AssertionError("Decoded beyond damaged packet")))
    install_container(monkeypatch, packets, codec)

    if case in {"start", "no_audio", "other_error"}:
        with pytest.raises(type(bad.error)):
            _decode_audio("broken.mpga")
        assert "Truncated audio" not in caplog.text
    else:
        signal, rate = _decode_audio("broken.mpga")
        assert rate == 44100
        np.testing.assert_array_equal(
            signal, np.full((1, 1152), 0.25, dtype=np.float32)
        )
        assert "Truncated audio" in caplog.text


def test_clean_packets_preserve_all_samples(monkeypatch, caplog):
    install_container(monkeypatch, [frame_packet(), frame_packet(), Packet(size=0)])

    signal, rate = _decode_audio("clean.mpga")

    assert rate == 44100
    assert signal.shape == (1, 2304)
    assert "Truncated audio" not in caplog.text


@pytest.mark.parametrize("has_audio", [False, True])
def test_demux_damage(monkeypatch, caplog, has_audio):
    def packets():
        if has_audio:
            yield frame_packet()
        raise av.error.InvalidDataError(1094995529, "Damaged container")

    install_container(monkeypatch, packets())
    if has_audio:
        signal, _ = _decode_audio("damaged.mp3")
        assert signal.shape == (1, 1152)
        assert "Truncated audio" in caplog.text
    else:
        with pytest.raises(av.error.InvalidDataError):
            _decode_audio("damaged.mp3")


def test_truncated_audio_load_succeeds(monkeypatch):
    install_container(monkeypatch, [frame_packet(), bad_packet(), frame_packet()])
    audio = Audio(device="cpu")
    signal, _ = audio.load("damaged.mp3")
    assert signal is not None
    assert audio.load_error is None
    assert len(signal) == int(np.ceil(1152 * audio.sampling_rate / 44100))


def test_non_utf8_metadata_preserves_audio(tmp_path):
    def chunk(name, data):
        return name + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)

    samples = np.full(1152, 8192, dtype="<i2")
    contents = (
        b"WAVE"
        + chunk(b"fmt ", struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16))
        + chunk(b"LIST", b"INFO" + chunk(b"INAM", b"Legacy \xa9 metadata\0"))
        + chunk(b"data", samples.tobytes())
    )
    path = tmp_path / "legacy.wav"
    path.write_bytes(chunk(b"RIFF", contents))

    with pytest.raises(UnicodeDecodeError):
        with av.open(str(path)):
            pass

    signal, rate = _decode_audio(str(path))

    assert rate == 44100
    np.testing.assert_array_equal(signal, np.full((1, 1152), 0.25, dtype=np.float32))
