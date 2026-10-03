import pickle
from copy import deepcopy

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from britekit.cli import cli
from britekit.commands import pickle_frame_infer
from britekit.core import config_loader, util
from britekit.core.base_config import BaseConfig
from britekit.core.dataset import SpectrogramDataset
from britekit.models import model_loader


class FrameModel:
    def __init__(self, cfg, codes, curves):
        self.settings = deepcopy(cfg)
        self.train_class_codes = codes
        self.curves = None if curves is None else np.asarray(curves, dtype=np.float32)
        self.inputs = []

    def eval(self):
        return self

    def to(self, device):
        assert device == "cpu"
        return self

    def apply_training_config(self, cfg):
        cfg.audio = deepcopy(self.settings.audio)
        cfg.train.sed_fps = self.settings.train.sed_fps
        self.cfg = cfg

    def predict(self, inputs, device):
        assert device == "cpu"
        self.inputs.append(inputs.copy())
        # Segment scores deliberately disagree with the frame curves.
        scores = np.ones((len(inputs), len(self.train_class_codes)), np.float32)
        frames = (
            None
            if self.curves is None
            else np.broadcast_to(self.curves, (len(inputs), *self.curves.shape)).copy()
        )
        return scores, frames


@pytest.fixture
def case(monkeypatch, tmp_path):
    monkeypatch.setattr(util, "get_device", lambda: "cpu")
    cfg = BaseConfig()
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 6
    cfg.audio.spec_duration = 3
    cfg.audio.convert_to_db = True
    cfg.train.sed_fps = 2
    cfg.train.augment = False
    cfg.infer.scaling_coefficient = 99
    cfg.infer.scaling_intercept = 99
    monkeypatch.setattr(config_loader, "_base_config", cfg)
    values = np.tile(np.array([1, 0.1, 0.01, 0.0001, 0, 0.001], np.float32), (2, 1))
    data = dict(
        class_names=["Alpha", "Beta"],
        class_codes=["a", "b"],
        spec_values=[util.compress_spectrogram(values, bits=16)] * 3,
        spec_segment_ids=[10, 20, 30],
        spec_class_indexes=[[0], [1], [0]],
    )
    source = tmp_path / "train.pkl"
    source.write_bytes(pickle.dumps(data))
    checkpoints = tmp_path / "ensemble"
    checkpoints.mkdir()
    return cfg, data, source, checkpoints, tmp_path / "frames.pkl"


def install_models(monkeypatch, checkpoints, models):
    paths = {}
    for i, model in enumerate(models):
        path = checkpoints / f"{i}.ckpt"
        path.write_bytes(b"checkpoint")
        paths[str(path)] = model

    def load(path, apply_training_config=True):
        assert not apply_training_config
        return paths[path]

    monkeypatch.setattr(model_loader, "load_from_checkpoint", load)


def test_cli_averages_known_class_curves_and_training_preserves_soft_labels(
    monkeypatch, case
):
    cfg, data, source, checkpoints, destination = case
    a1 = [0.1, 0.8, 0.05, 0.05, 0.7, 0.1]
    b1 = [0.2, 0.3, 0.6, 0.9, 0.2, 0.1]
    a2 = [0.3, 0.6, 0.15, 0.15, 0.9, 0.3]
    b2 = [0.4, 0.5, 0.8, 0.7, 0.4, 0.3]
    first = FrameModel(cfg, ["a", "b", "extra"], [a1, b1, [1] * 6])
    second = FrameModel(cfg, ["b", "a"], [b2, a2])
    second.settings.audio.top_db = 40
    second.settings.audio.db_power = 2
    install_models(monkeypatch, checkpoints, [first, second])
    original_cfg = deepcopy(cfg)

    result = CliRunner().invoke(
        cli,
        [
            "pickle-frame-infer",
            str(source),
            "--checkpoints",
            str(checkpoints),
            "--output",
            str(destination),
            "--batch-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    labels = pickle.loads(destination.read_bytes())
    assert list(labels) == [10, 20, 30]
    expected = [np.mean([a1, a2], axis=0), np.mean([b1, b2], axis=0)]
    for segment_id, curve in labels.items():
        assert type(segment_id) is int
        assert curve.dtype == np.float32 and curve.shape == (6,)
        np.testing.assert_allclose(curve, expected[segment_id == 20], atol=1e-7)
    # Two separated calls retain the low-probability gap and are not peak-normalized.
    assert labels[10][2] == pytest.approx(0.1)
    assert labels[10].max() == pytest.approx(0.8)

    from britekit.core.audio_util import convert_to_db

    expanded = util.expand_spectrogram(data["spec_values"][0], cfg=cfg)
    for model in (first, second):
        settings = model.settings.audio
        expected_input = convert_to_db(
            expanded, settings.power, settings.top_db, db_power=settings.db_power
        )
        assert [len(batch) for batch in model.inputs] == [2, 1]
        np.testing.assert_allclose(model.inputs[0][0], expected_input)
        assert model.cfg.infer.scaling_coefficient == 1
        assert model.cfg.infer.scaling_intercept == 0
    assert cfg == original_cfg
    assert source.read_bytes() == pickle.dumps(data)

    dataset = SpectrogramDataset(
        data["spec_values"],
        data["spec_class_indexes"],
        2,
        segment_ids=data["spec_segment_ids"],
        frame_label_dict=labels,
    )
    for i, class_index in enumerate([0, 1, 0]):
        item = dataset[i]
        np.testing.assert_allclose(
            item["frame_labels"][:, class_index], expected[class_index], atol=1e-7
        )
        assert not item["frame_labels"][:, 1 - class_index].any()
        torch.testing.assert_close(item["segment_labels"], torch.eye(2)[class_index])
        assert "teacher_segment_labels" not in item


@pytest.mark.parametrize("use_cli", [False, True])
def test_full_segment_classes_override_only_labeled_segments(
    monkeypatch, case, caplog, use_cli
):
    cfg, data, source, checkpoints, destination = case
    data.update(
        class_names=["Noise", "Insects", "Other", "Alpha"],
        class_codes=["n", "i", "o", "a"],
        spec_values=data["spec_values"] * 2,
        spec_segment_ids=[10, 20, 30, 40, 50, 60],
        spec_class_indexes=[[0], [3], [1], [3], [2], [0]],
    )
    source.write_bytes(pickle.dumps(data))
    a1 = [0.1, 0.8, 0.05, 0.05, 0.7, 0.1]
    a2 = [0.3, 0.6, 0.15, 0.15, 0.9, 0.3]
    first = FrameModel(cfg, ["n", "i", "o", "a"], [[0.2] * 6] * 3 + [a1])
    second = FrameModel(cfg, ["a", "o", "i", "n"], [a2] + [[0.4] * 6] * 3)
    install_models(monkeypatch, checkpoints, [first, second])

    with caplog.at_level("INFO"):
        if use_cli:
            result = CliRunner().invoke(
                cli,
                [
                    "pickle-frame-infer",
                    str(source),
                    "--checkpoints",
                    str(checkpoints),
                    "--output",
                    str(destination),
                    "--batch-size",
                    "2",
                    "--full-segment-classes",
                    "Noise, Insects, Other",
                ],
            )
            assert result.exit_code == 0, result.output
        else:
            pickle_frame_infer(
                str(source),
                str(checkpoints),
                str(destination),
                batch_size=2,
                full_segment_classes=["Noise", "Insects", "Other"],
            )

    labels = pickle.loads(destination.read_bytes())
    assert list(labels) == data["spec_segment_ids"]
    dataset = SpectrogramDataset(
        data["spec_values"],
        data["spec_class_indexes"],
        4,
        segment_ids=data["spec_segment_ids"],
        frame_label_dict=labels,
    )
    for i, (segment_id, indexes) in enumerate(
        zip(data["spec_segment_ids"], data["spec_class_indexes"])
    ):
        class_index = indexes[0]
        expected = np.mean([a1, a2], axis=0) if class_index == 3 else np.ones(6)
        curve = labels[segment_id]
        assert curve.dtype == np.float32 and curve.shape == (6,)
        np.testing.assert_allclose(curve, expected, atol=1e-7)
        item = dataset[i]
        expected_frames = np.zeros((6, 4), dtype=np.float32)
        expected_frames[:, class_index] = expected
        np.testing.assert_allclose(item["frame_labels"], expected_frames, atol=1e-7)
        torch.testing.assert_close(item["segment_labels"], torch.eye(4)[class_index])
    log_output = result.output if use_cli else caplog.text
    assert "Forcing all-one frame labels for 4 segments" in log_output
    assert source.read_bytes() == pickle.dumps(data)


@pytest.mark.parametrize(
    "problem, message",
    [
        ("unknown", "Unknown full-segment class names.*Noise"),
        ("case", "Unknown full-segment class names.*alpha"),
        ("missing_names", "missing required keys.*class_names"),
        ("name_count", "class_names matching class_codes"),
        ("duplicate_names", "unique, nonempty class_names"),
        ("empty_name", "list of nonempty class names"),
        ("bare_string", "list of nonempty class names"),
    ],
)
def test_invalid_full_segment_classes_rejected_before_inference(
    monkeypatch, case, problem, message
):
    _, data, source, checkpoints, destination = case
    classes = ["Alpha"]
    if problem == "unknown":
        classes = ["Noise"]
    elif problem == "case":
        classes = ["alpha"]
    elif problem == "missing_names":
        del data["class_names"]
    elif problem == "name_count":
        data["class_names"].pop()
    elif problem == "duplicate_names":
        data["class_names"] = ["Alpha", "Alpha"]
    elif problem == "empty_name":
        classes = [""]
    elif problem == "bare_string":
        classes = "Alpha"
    source.write_bytes(pickle.dumps(data))
    destination.write_bytes(b"previous labels")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid class overrides should be rejected before loading models")

    monkeypatch.setattr(model_loader, "load_from_checkpoint", unexpected_load)
    with pytest.raises(ValueError, match=message):
        pickle_frame_infer(
            str(source),
            str(checkpoints),
            str(destination),
            full_segment_classes=classes,
        )
    assert destination.read_bytes() == b"previous labels"


@pytest.mark.parametrize(
    "problem, message",
    [
        ("multi_label", "exactly one class"),
        ("unlabeled", "exactly one class"),
        ("class_index", "Invalid class index"),
        ("missing_labels", "spec_class_indexes"),
        ("label_count", "numbers of labels"),
        ("duplicates", "duplicate segment IDs"),
        ("empty", "no spectrograms"),
    ],
)
def test_invalid_input_is_rejected_before_inference(
    monkeypatch, case, problem, message
):
    cfg, data, source, checkpoints, destination = case
    if problem == "multi_label":
        data["spec_class_indexes"][0] = [0, 1]
    elif problem == "unlabeled":
        data["spec_class_indexes"][0] = []
    elif problem == "class_index":
        data["spec_class_indexes"][0] = [2]
    elif problem == "missing_labels":
        del data["spec_class_indexes"]
    elif problem == "label_count":
        data["spec_class_indexes"].pop()
    elif problem == "duplicates":
        data["spec_segment_ids"][1] = 10
    elif problem == "empty":
        data["spec_values"] = data["spec_segment_ids"] = data["spec_class_indexes"] = []
    source.write_bytes(pickle.dumps(data))
    destination.write_bytes(b"previous labels")
    with pytest.raises(ValueError, match=message):
        pickle_frame_infer(str(source), str(checkpoints), str(destination))
    assert destination.read_bytes() == b"previous labels"


@pytest.mark.parametrize(
    "problem, message",
    [
        ("classification_only", "SED frame outputs"),
        ("missing_class", "missing.*b"),
        ("duration", "spec_duration"),
        ("fps", "matching sed_fps"),
        ("shape", "Unexpected frame output shape"),
        ("nan", "finite probabilities"),
        ("out_of_range", "finite probabilities"),
    ],
)
def test_invalid_ensemble_does_not_replace_output(monkeypatch, case, problem, message):
    cfg, _, source, checkpoints, destination = case
    first = FrameModel(cfg, ["a", "b"], [[0.2] * 6, [0.4] * 6])
    second = FrameModel(cfg, ["a", "b"], [[0.6] * 6, [0.8] * 6])
    if problem == "classification_only":
        second.curves = None
    elif problem == "missing_class":
        second.train_class_codes = ["a", "c"]
    elif problem == "duration":
        second.settings.audio.spec_duration = 4
    elif problem == "fps":
        second.settings.train.sed_fps = 4
    elif problem == "shape":
        second.curves = second.curves[:, :-1]
    elif problem == "nan":
        second.curves[0, 0] = np.nan
    elif problem == "out_of_range":
        second.curves[0, 0] = 1.01
    install_models(monkeypatch, checkpoints, [first, second])
    destination.write_bytes(b"previous labels")
    with pytest.raises(ValueError, match=message):
        pickle_frame_infer(str(source), str(checkpoints), str(destination))
    assert destination.read_bytes() == b"previous labels"


def test_output_cannot_overwrite_input(monkeypatch, case):
    cfg, _, source, checkpoints, _ = case
    install_models(
        monkeypatch, checkpoints, [FrameModel(cfg, ["a", "b"], [[0.2] * 6] * 2)]
    )
    original = source.read_bytes()
    with pytest.raises(ValueError, match="Output path must differ"):
        pickle_frame_infer(str(source), str(checkpoints), str(source))
    assert source.read_bytes() == original


@pytest.mark.parametrize("pooling", ["logsumexp", "linear_softmax"])
def test_real_bknet_convnext_ensemble_matches_direct_inference(
    monkeypatch, tmp_path, pooling
):
    import lightning.pytorch as pl

    checkpoints = tmp_path / "ensemble"
    checkpoints.mkdir()
    specs = [
        util.compress_spectrogram(
            np.geomspace(0.0001, 1, 32 * 64).astype(np.float32).reshape(32, 64)
        )
    ]
    expected = []
    monkeypatch.setattr(model_loader, "get_device", lambda: "cpu")
    monkeypatch.setattr(util, "get_device", lambda: "cpu")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for model_type, top_db in [("bknet.1", 80), ("timm.convnextv2_femto", 40)]:
            cfg = BaseConfig()
            cfg.audio.spec_height, cfg.audio.spec_width = 32, 64
            cfg.audio.spec_duration = 3
            cfg.audio.convert_to_db = True
            cfg.audio.top_db = top_db
            cfg.train.model_type = model_type
            cfg.train.head_type = "temporal_sed"
            cfg.train.temporal_pooling = pooling
            cfg.train.hidden_channels = 8
            cfg.train.pretrained = False
            cfg.train.frame_loss_weight = 0
            monkeypatch.setattr(config_loader, "_base_config", cfg)
            model = model_loader.load_new_model(
                ["Alpha", "Beta"], ["a", "b"], ["", ""], ["", ""], 1
            ).eval()
            dataset = SpectrogramDataset(specs, [[1]], 2, is_training=False)
            expected.append(
                model.predict(dataset[0]["input"].unsqueeze(0), "cpu")[1][0, 1]
            )
            state = dict(
                epoch=0,
                state_dict=model.state_dict(),
                hyper_parameters=dict(model.hparams),
                **{"pytorch-lightning_version": pl.__version__},
            )
            model.on_save_checkpoint(state)
            torch.save(state, checkpoints / f"{model_type}.ckpt")
        source = tmp_path / "train.pkl"
        source.write_bytes(
            pickle.dumps(
                dict(
                    class_codes=["a", "b"],
                    spec_values=specs,
                    spec_segment_ids=[42],
                    spec_class_indexes=[[1]],
                )
            )
        )
        destination = tmp_path / "frames.pkl"
        runtime_cfg = BaseConfig()
        monkeypatch.setattr(config_loader, "_base_config", runtime_cfg)
        pickle_frame_infer(str(source), str(checkpoints), str(destination))
        labels = pickle.loads(destination.read_bytes())
        assert list(labels) == [42]
        assert labels[42].shape == (12,)
        np.testing.assert_allclose(labels[42], np.mean(expected, axis=0), atol=1e-6)
        assert runtime_cfg == BaseConfig()
    finally:
        torch.set_num_threads(previous_threads)
