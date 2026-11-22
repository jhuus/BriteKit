from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from britekit.core import predictor
from britekit.core.base_config import BaseConfig
from britekit.models.base_model import BaseModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("block_size, expected_calls", [(200, 2), (1, 4)])
def test_overlapping_input_reuse_is_recording_local(
    tmp_path, monkeypatch, block_size, expected_calls
):
    class RecordingModel(BaseModel):
        def __init__(self, cfg):
            torch.nn.Module.__init__(self)
            self.cfg = cfg
            self.inputs = []
            self.eval()

        def predict(self, x, device=None, preprocess=None):
            tensor = self._ensure_tensor(x, device)
            self.inputs.append(tensor)
            scores = tensor.mean(dim=(1, 2, 3)).cpu().numpy()
            return scores[:, None], None

    p = predictor.Predictor.__new__(predictor.Predictor)
    p.cfg = BaseConfig()
    p.cfg.audio.spec_duration = 3
    p.cfg.infer.block_size = block_size
    p.cfg.infer.audio_power = 0.7
    p.device = "cuda"
    p.ov = None
    p.models = [RecordingModel(p.cfg) for _ in range(4)]
    recordings = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for recording in recordings:
        recording.touch()
    p.audio = SimpleNamespace(seconds=lambda: 7.0)
    p.audio.load = Mock()

    def spectrogram(starts):
        gain = 0.1 if p.audio.load.call_args.args[0] == str(recordings[0]) else 0.2
        values = np.broadcast_to(
            (gain + 0.01 * np.asarray(starts, dtype=np.float32))[:, None, None],
            (len(starts), 2, 3),
        ).copy()
        return values, values * 2

    p.audio.get_spectrograms = Mock(side_effect=spectrogram)

    actual = [p.get_overlapping_scores(str(path), [0, 1]) for path in recordings]
    assert p.audio.get_spectrograms.call_count == 2 * expected_calls
    for run in range(2):
        inputs = [m.inputs[run] for m in p.models]
        assert (inputs[0] is inputs[2]) == (block_size == 200)
        assert (inputs[1] is inputs[3]) == (block_size == 200)
        assert inputs[0] is not inputs[1]
    assert p.models[0].inputs[0] is not p.models[0].inputs[1]

    monkeypatch.setattr(predictor, "_OVERLAPPING_INPUT_CACHE_BYTES", 0)
    expected = [p.get_overlapping_scores(str(path), [0, 1]) for path in recordings]
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)
    assert p.audio.get_spectrograms.call_count == 2 * expected_calls + 8
