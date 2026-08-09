import random
from types import SimpleNamespace

import pytest

from britekit.core.data_module import DataModule, PerRecordingSampler


def test_sampler_caps_common_classes_and_retains_rare_classes():
    # Class 0 occurs in three recordings; class 1 occurs in only one.
    recording_ids = [10, 10, 10, 10, 20, 30]
    class_indexes = [[0], [0], [1], [1], [0], [0]]
    sampler = PerRecordingSampler(
        list(range(6)),
        recording_ids,
        class_indexes,
        max_per_recording=1,
        min_recordings=2,
    )

    random.seed(1)
    selected = list(sampler)

    assert {2, 3}.issubset(selected)
    assert len(set(selected).intersection({0, 1})) == 1
    assert {4, 5}.issubset(selected)
    assert len(sampler) == 5


def test_sampler_protects_multilabel_samples_with_a_rare_class():
    recording_ids = [10, 10, 20, 30]
    class_indexes = [[0], [0, 1], [0], [0]]
    sampler = PerRecordingSampler(
        list(range(4)),
        recording_ids,
        class_indexes,
        max_per_recording=1,
        min_recordings=2,
    )

    selected = list(sampler)

    assert 1 in selected
    assert len(set(selected).intersection({0, 1})) == 2


@pytest.mark.parametrize(
    ("max_per_recording", "min_recordings"), [(0, None), (-1, None), (1, 0)]
)
def test_sampler_rejects_nonpositive_limits(max_per_recording, min_recordings):
    with pytest.raises(ValueError, match="must be positive"):
        PerRecordingSampler(
            [0],
            [10],
            [[0]],
            max_per_recording=max_per_recording,
            min_recordings=min_recordings,
        )


def test_data_module_rejects_missing_recording_ids():
    module = DataModule.__new__(DataModule)
    module.train_data = object()
    module.recording_ids = None
    module.cfg = SimpleNamespace(train=SimpleNamespace(max_per_recording=1))

    with pytest.raises(ValueError, match="requires.*recording IDs"):
        module.train_dataloader()
