from types import SimpleNamespace

import numpy as np

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
