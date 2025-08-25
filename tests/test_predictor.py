import os
from pathlib import Path

from britekit import Predictor
from britekit.core.config_loader import get_config

cfg, _ = get_config()
recording_dir = str(Path("tests") / "recordings")
label_dir = str(Path("tests") / "labels")

if not os.path.exists(label_dir):
    os.makedirs(label_dir)


def test_basic_operation():
    model_path = str(Path("tests") / "ckpt" / "test1.ckpt")
    predictor = Predictor(model_path)

    recording_path = str(Path(recording_dir) / "CommonYellowthroat.mp3")
    preds, _, offsets = predictor.get_raw_scores(recording_path)
    assert preds.shape == (2, 6)

    cfg.infer.min_score = 0.4
    file_path = str(Path(label_dir) / "CommonYellowthroat_labels.txt")
    predictor.save_audacity_labels(preds, None, offsets, file_path)
