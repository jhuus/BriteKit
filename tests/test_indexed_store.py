import json
import pickle
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from britekit.core import config_loader
from britekit.core.base_config import BaseConfig
from britekit.core.convert_pickles import convert_pickle
from britekit.core.data_module import DataModule
from britekit.core.indexed_store import IndexView, read_store
from britekit.core.util import compress_spectrogram


@pytest.fixture
def stores(tmp_path):
    cfg = BaseConfig()
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 3
    cfg.audio.spec_duration = 1
    cfg.audio.convert_to_db = False
    cfg.train.sed_fps = 3
    cfg.train.augment = False
    cfg.train.batch_size = 2
    cfg.train.num_workers = 0
    cfg.train.val_portion = 0.5
    cfg.train.noise_class_name = "Noise"
    cfg.train.teacher_only_if_no_frame = True
    training = dict(
        class_names=["Bird", "Noise"],
        class_codes=["B", "N"],
        alt_names=["Bird", "Noise"],
        alt_codes=["B", "N"],
        spec_values=[
            compress_spectrogram(
                np.full((2, 3), i / 4, dtype=np.float32), bits=8 if i % 2 else 16
            )
            for i in range(4)
        ],
        spec_class_indexes=[[0], [1], [0, 1], [0]],
        spec_segment_ids=[10, 20, 30, 40],
        spec_recording_ids=[1, 1, 2, 3],
    )
    teacher = dict(
        format_version=2,
        class_codes=["B", "N"],
        segment_ids=[30, 10, 40],
        probabilities=np.array([[0.3, 0.4], [0.1, 0.2], [0.8, 0.9]], dtype=np.float32),
        frame_probabilities=np.arange(18, dtype=np.float16).reshape(3, 2, 3) / 32,
    )
    frames = {
        10: np.array([0, 1, 0], dtype=np.float32),
        40: np.zeros(3, dtype=np.float32),
    }
    for kind, data in [
        ("training", training),
        ("teacher-targets", teacher),
        ("training-frames", frames),
    ]:
        source = tmp_path / f"{kind}.pkl"
        source.write_bytes(pickle.dumps(data))
        convert_pickle(kind, str(source), str(tmp_path / kind))
    previous = config_loader.get_config()
    yield cfg, tmp_path
    config_loader.set_base_config(previous)


def module(cfg, root, indexed=True):
    cfg = deepcopy(cfg)
    suffix = "" if indexed else ".pkl"
    cfg.train.train_pickle = str(root / f"training{suffix}")
    cfg.train.test_pickle = cfg.train.train_pickle
    cfg.train.teacher_targets_pickle = str(root / f"teacher-targets{suffix}")
    cfg.train.frame_label_pickle = str(root / f"training-frames{suffix}")
    config_loader.set_base_config(cfg)
    result = DataModule()
    result.prepare_fold(0)
    return result


def test_all_sample_outputs_match_legacy_with_missing_reordered_teacher_rows(stores):
    cfg, root = stores
    legacy = module(cfg, root, False)
    indexed = module(cfg, root)
    for i in range(4):
        a, b = legacy.full_dataset[i], indexed.full_dataset[i]
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key])
    assert indexed.full_dataset.teacher_target_mask.tolist() == [
        True,
        False,
        True,
        True,
    ]
    assert isinstance(indexed.val_data.dataset.compressed_specs, IndexView)
    assert indexed.full_dataset._has_frame_label(
        3
    )  # present all-zero frame supervision
    assert not indexed.full_dataset._has_frame_label(1)
    for method in ("train_dataloader", "val_dataloader", "test_dataloader"):
        batch = next(iter(getattr(indexed, method)()))
        assert batch["input"].shape == (2, 1, 2, 3)


def test_spawn_worker_and_serialization_do_not_embed_mapped_payloads(stores):
    cfg, root = stores
    indexed = module(cfg, root)
    expected = indexed.full_dataset[0]
    serialized = pickle.dumps(indexed.full_dataset)
    assert len(serialized) < 30000
    restored = pickle.loads(serialized)
    assert restored.compressed_specs._data is None
    assert restored.teacher_targets.source._array is None
    for key, value in expected.items():
        torch.testing.assert_close(restored[0][key], value)
    loader = DataLoader(
        indexed.full_dataset,
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )
    for _ in range(2):
        batches = list(loader)
        torch.testing.assert_close(batches[0]["input"][0], expected["input"])
    del loader


def test_validation_view_does_not_read_spectrograms(stores, monkeypatch):
    cfg, root = stores
    indexed = module(cfg, root)

    def unexpected(*args):
        raise AssertionError("Split fetched a spectrogram")

    monkeypatch.setattr(type(indexed.specs), "__getitem__", unexpected)
    dataset = indexed._make_val_dataset([1, 3])
    assert len(dataset) == 2


def test_repeated_reads_are_writable_and_do_not_change_store(stores):
    cfg, root = stores
    indexed = module(cfg, root)
    expected = indexed.full_dataset._get_spec(0)
    modified = indexed.full_dataset._get_spec(0)
    modified[:] = 99
    np.testing.assert_array_equal(indexed.full_dataset._get_spec(0), expected)
    row = indexed.full_dataset.teacher_targets[0]
    row[:] = 99
    assert (indexed.full_dataset.teacher_targets[0] <= 1).all()


def test_kfold_and_recording_sampler(stores):
    cfg, root = stores
    cfg.train.num_folds = 2
    cfg.train.max_per_recording = 1
    cfg.train.val_max_per_recording = 1
    indexed = module(cfg, root)
    for fold in range(2):
        indexed.prepare_fold(fold)
        assert next(iter(indexed.train_dataloader()))["input"].shape[-2:] == (2, 3)
        assert next(iter(indexed.val_dataloader()))["input"].shape[-2:] == (2, 3)


@pytest.mark.parametrize("damage", ["version", "offset", "shape"])
def test_reject_malformed_store(stores, damage):
    _, root = stores
    store = root / "training"
    if damage == "version":
        path = store / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["format_version"] = 99
        path.write_text(json.dumps(manifest))
    elif damage == "offset":
        np.save(store / "offsets.npy", np.array([0, 100, 1, 2, 3], dtype=np.uint64))
    else:
        np.save(store / "segment_ids.npy", np.array([1]))
    with pytest.raises(ValueError):
        read_store(str(store), "training")


def test_train_cli_accepts_store_directories(stores, monkeypatch):
    from click.testing import CliRunner
    from britekit.commands import _train

    _, root = stores
    received = []
    monkeypatch.setattr(_train, "train", lambda *args: received.append(args))
    result = CliRunner().invoke(
        _train._train_cmd,
        [
            "--teacher-targets",
            str(root / "teacher-targets"),
            "--frame-labels",
            str(root / "training-frames"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert received[0][-2:] == (
        str(root / "training-frames"),
        str(root / "teacher-targets"),
    )


def test_teacher_class_order_mismatch_fails(stores):
    cfg, root = stores
    path = root / "teacher-targets/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"]["class_codes"].reverse()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="class codes"):
        module(cfg, root)
