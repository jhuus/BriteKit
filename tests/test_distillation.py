import pickle
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.core.data_module import DataModule
from britekit.core.dataset import SpectrogramDataset
from britekit.core.util import compress_spectrogram
from britekit.models.base_model import BaseModel


@pytest.fixture
def distillation_cfg():
    cfg = BaseConfig()
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 4
    cfg.audio.spec_duration = 2
    cfg.train.sed_fps = 1
    cfg.train.frame_loss_weight = 0.1
    cfg.train.distillation_weight = 0.5
    cfg.train.distillation_temperature = 1.0
    set_base_config(cfg)
    yield cfg
    set_base_config(BaseConfig())


def _model(cfg):
    model = BaseModel(
        model_type="test",
        head_type=None,
        hidden_channels=1,
        train_class_names=["A"],
        train_class_codes=["a"],
        train_class_alt_names=["A"],
        train_class_alt_codes=["a"],
        num_train_specs=1,
        multi_label=True,
    )
    model.cfg = cfg
    model.use_sed = True
    return model


def test_distillation_blends_only_the_segment_loss(distillation_cfg):
    model = _model(distillation_cfg)
    segment_logits = torch.tensor([[2.0]])
    frame_logits = torch.zeros((1, 1, 2))
    hard_segment = torch.ones((1, 1))
    hard_frames = torch.ones((1, 2, 1))
    teacher = torch.zeros((1, 1))

    loss = model._calc_loss(
        segment_logits,
        frame_logits,
        hard_segment,
        hard_segment,
        hard_frames,
        teacher,
    )

    hard_loss = F.binary_cross_entropy_with_logits(segment_logits, hard_segment)
    teacher_loss = F.binary_cross_entropy_with_logits(segment_logits, teacher)
    frame_loss = F.binary_cross_entropy_with_logits(
        frame_logits.transpose(1, 2), hard_frames
    )
    expected = 0.9 * (0.5 * hard_loss + 0.5 * teacher_loss) + 0.1 * frame_loss
    torch.testing.assert_close(loss, expected)


def test_missing_frame_sample_uses_teacher_loss_only(distillation_cfg):
    model = _model(distillation_cfg)
    segment_logits = torch.tensor([[2.0], [2.0]])
    frame_logits = torch.zeros((2, 1, 2))
    hard_segment = torch.ones((2, 1))
    hard_frames = torch.ones((2, 2, 1))
    teacher = torch.zeros((2, 1))
    teacher_frames = torch.zeros((2, 2, 1))

    loss = model._calc_loss(
        segment_logits,
        frame_logits,
        hard_segment,
        hard_segment,
        hard_frames,
        teacher,
        teacher_frames,
        hard_segment_mask=torch.tensor([1.0, 0.0]),
        frame_label_mask=torch.tensor([1.0, 0.0]),
    )

    hard_loss = F.binary_cross_entropy_with_logits(segment_logits[0], hard_segment[0])
    teacher_loss = F.binary_cross_entropy_with_logits(segment_logits[0], teacher[0])
    frame_loss = F.binary_cross_entropy_with_logits(
        frame_logits[0].transpose(0, 1), hard_frames[0]
    )
    original_loss = 0.9 * (0.5 * hard_loss + 0.5 * teacher_loss) + 0.1 * frame_loss
    teacher_frame_loss = F.binary_cross_entropy_with_logits(
        frame_logits[1].transpose(0, 1), teacher_frames[1]
    )
    new_loss = 0.9 * teacher_loss + 0.1 * teacher_frame_loss
    expected = (original_loss + new_loss) / 2
    torch.testing.assert_close(loss, expected)


def test_teacher_only_masks_follow_frame_label_membership(distillation_cfg):
    cfg = distillation_cfg
    cfg.train.augment = False
    cfg.train.teacher_only_if_no_frame = True
    specs = [
        compress_spectrogram(np.ones((2, 4), dtype=np.float32)),
        compress_spectrogram(np.ones((2, 4), dtype=np.float32)),
    ]
    dataset = SpectrogramDataset(
        specs,
        [[0], [0]],
        1,
        segment_ids=[10, 20],
        frame_label_dict={10: np.array([1, 0], dtype=np.float32)},
        teacher_targets=np.array([[0.8], [0.2]], dtype=np.float32),
        teacher_frame_targets=np.array([[[0.8, 0.1]], [[0.2, 0.3]]], dtype=np.float32),
    )

    original = dataset[0]
    random_segment = dataset[1]

    assert original["hard_segment_mask"].item() == 1
    assert original["frame_label_mask"].item() == 1
    assert random_segment["hard_segment_mask"].item() == 0
    assert random_segment["frame_label_mask"].item() == 0
    assert random_segment["segment_labels"].item() == 1
    assert random_segment["teacher_segment_labels"].item() == pytest.approx(0.2)
    np.testing.assert_allclose(
        random_segment["teacher_frame_labels"].numpy()[:, 0], [0.2, 0.3]
    )


def test_teacher_targets_follow_simple_merge(distillation_cfg):
    cfg = distillation_cfg
    cfg.train.augment = True
    cfg.train.multi_label = True
    cfg.train.prob_simple_merge = 1.0
    cfg.train.prob_mixup = 0.0
    cfg.train.prob_cutmix = 0.0
    cfg.train.prob_fade1 = 0.0
    cfg.train.augmentations = []
    specs = [
        compress_spectrogram(np.ones((2, 4), dtype=np.float32)),
        compress_spectrogram(np.ones((2, 4), dtype=np.float32)),
    ]
    dataset = SpectrogramDataset(
        specs,
        [[0], [1]],
        2,
        teacher_targets=np.array([[0.8, 0.1], [0.2, 0.7]], dtype=np.float32),
        teacher_frame_targets=np.array(
            [
                [[0.8, 0.4], [0.1, 0.2]],
                [[0.2, 0.5], [0.7, 0.6]],
            ],
            dtype=np.float32,
        ),
    )

    with (
        patch("britekit.core.dataset.random.random", return_value=0.0),
        patch("britekit.core.dataset.random.randint", return_value=1),
    ):
        item = dataset[0]

    torch.testing.assert_close(
        item["teacher_segment_labels"], torch.tensor([0.84, 0.73])
    )
    torch.testing.assert_close(
        item["teacher_frame_labels"],
        torch.tensor([[0.84, 0.73], [0.70, 0.68]]),
    )


def test_teacher_targets_are_reordered_by_segment_id(tmp_path):
    path = tmp_path / "teacher.pkl"
    with path.open("wb") as file:
        pickle.dump(
            {
                "format_version": 2,
                "class_codes": ["a", "b"],
                "segment_ids": [20, 10],
                "probabilities": np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32),
                "frame_probabilities": np.array(
                    [
                        [[0.2, 0.3], [0.8, 0.7]],
                        [[0.9, 0.8], [0.1, 0.2]],
                    ],
                    dtype=np.float16,
                ),
            },
            file,
        )

    module = DataModule.__new__(DataModule)
    targets, frame_targets = module._load_teacher_targets(path, ["a", "b"], [10, 20])

    np.testing.assert_array_equal(
        targets, np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        frame_targets,
        np.array(
            [
                [[0.9, 0.8], [0.1, 0.2]],
                [[0.2, 0.3], [0.8, 0.7]],
            ],
            dtype=np.float32,
        ),
        atol=0.001,
    )
