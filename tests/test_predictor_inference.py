from types import SimpleNamespace

import numpy as np
import pytest

from britekit.core.predictor import Predictor


def test_to_global_frames_averages_overlaps_and_clips_edges():
    predictor = Predictor.__new__(Predictor)
    predictor.cfg = SimpleNamespace(train=SimpleNamespace(sed_fps=2.0))
    frame_scores = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[10.0, 20.0, 30.0, 40.0]],
        ],
        dtype=np.float32,
    )

    result = predictor.to_global_frames(
        frame_scores, offsets_sec=[-0.5, 0.5], recording_duration_sec=3.0
    )

    np.testing.assert_array_equal(
        result[:, 0], np.array([2.0, 6.5, 12.0, 30.0, 40.0, 0.0])
    )


@pytest.mark.parametrize(
    "offsets,duration",
    [
        ([-1.0, 1.0, 3.0], 4.5),  # clip both ends
        ([0.5, 2.5, 4.5], 7.0),  # uncovered leading and trailing frames
        ([-9.0, -7.0, -5.0], 4.5),  # entirely outside the recording
        ([0.0, 2.5, 4.5], 7.0),  # non-overlapping with a gap
    ],
)
def test_to_global_frames_matches_general_accumulator(offsets, duration):
    predictor = Predictor.__new__(Predictor)
    predictor.cfg = SimpleNamespace(train=SimpleNamespace(sed_fps=2.0))
    scores = np.arange(24, dtype=np.float32).reshape(3, 4, 2).transpose(0, 2, 1)
    scores[0, 0, 2] = -0.0  # Clipping should still include this frame.
    # Float64 input exercises the existing accumulator with the same values.
    expected = predictor.to_global_frames(scores.astype(np.float64), offsets, duration)
    actual = predictor.to_global_frames(scores, offsets, duration)
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("duration", [6.0, 6.5, 2.75])
def test_overlapping_scores_fill_only_missing_tail(tmp_path, duration):
    recording = tmp_path / "recording.wav"
    recording.touch()
    windows = []
    batches = []

    def get_spectrograms(offsets):
        windows.append(list(offsets))
        specs = np.ones((len(offsets), 2, 2), dtype=np.float32)
        return specs, specs

    def predict(specs, device):
        batches.append(len(specs))
        frames = np.arange(12, dtype=np.float32) / 32
        if len(batches) > 1:
            frames = frames + 0.5
        return None, np.broadcast_to(frames, (len(specs), 1, 12))

    predictor = Predictor.__new__(Predictor)
    predictor.cfg = SimpleNamespace(
        audio=SimpleNamespace(spec_duration=3.0),
        train=SimpleNamespace(sed_fps=4),
        infer=SimpleNamespace(audio_power=1.0),
    )
    predictor.ov = None
    predictor.device = "cpu"
    predictor.models = [SimpleNamespace(predict=predict)]
    predictor.audio = SimpleNamespace(
        load=lambda path: None,
        seconds=lambda: duration,
        get_spectrograms=get_spectrograms,
    )

    result = predictor.get_overlapping_scores(str(recording), [0])
    normal = np.arange(12, dtype=np.float32) / 32
    if duration == 6.5:
        assert windows == [[0, 3], [3.5]]
        assert batches == [2, 1]
        np.testing.assert_array_equal(result[:24, 0], np.tile(normal, 2))
        np.testing.assert_array_equal(result[24:, 0], normal[-2:] + 0.5)
        assert len(predictor.normalized_specs) == 2
    else:
        assert len(windows) == len(batches) == 1
        np.testing.assert_array_equal(
            result[:, 0], np.tile(normal, batches[0])[: round(duration * 4)]
        )


def test_get_recording_scores_can_reuse_loaded_audio(tmp_path):
    recording = tmp_path / "recording.wav"
    recording.touch()
    specs = np.ones((1, 2, 2), dtype=np.float32)

    class LoadedAudio:
        signal = np.ones(10, dtype=np.float32)

        def load(self, path):
            raise AssertionError("loaded audio should be reused")

        def seconds(self):
            return 1.0

        def get_spectrograms(self, start_times):
            return specs, specs

    predictor = Predictor.__new__(Predictor)
    predictor.audio = LoadedAudio()
    predictor.cfg = SimpleNamespace(
        audio=SimpleNamespace(spec_duration=1.0),
        infer=SimpleNamespace(audio_power=1.0),
    )
    predictor.get_start_times = lambda *args, **kwargs: [0.0]
    predictor.get_block_scores = lambda values, starts: (values, values)

    segment_scores, frame_map, start_times = predictor.get_recording_scores(
        str(recording), use_loaded_audio=True
    )

    np.testing.assert_array_equal(segment_scores, specs[:, None, :, :])
    np.testing.assert_array_equal(frame_map, specs[:, None, :, :])
    assert start_times == [0.0]
