import torch

from britekit.models.bknet import MODEL_REGISTRY
from britekit.models.bknet_base import BKNetBaseModel, ConvBnAct, ThickStem


def test_bknet_5_retains_standard_stem_by_default():
    model = BKNetBaseModel(MODEL_REGISTRY["bknet.5"], final_ch=1920)

    assert isinstance(model.stem, ConvBnAct)
    assert not isinstance(model.stem, ThickStem)
    assert model.stem.conv.out_channels == 24
    assert len(model.stages[0]) == 1

    output = model(torch.zeros(1, 1, 192, 384))
    assert output.shape == (1, 1920, 6, 12)


def test_bknet_5t_uses_thick_stem_in_place_of_stage0():
    model = BKNetBaseModel(MODEL_REGISTRY["bknet.5t"], final_ch=1920)

    input = torch.zeros(1, 1, 192, 384)
    stem_output = model.stem(input)
    output = model(input)

    assert isinstance(model.stem, ThickStem)
    assert stem_output.shape == (1, 64, 48, 96)
    assert isinstance(model.stages[0], torch.nn.Identity)
    assert model.stages[1][0].conv1.conv.in_channels == 64
    assert output.shape == (1, 1920, 6, 12)


def test_every_bknet_config_has_a_scaled_thick_stem_counterpart():
    assert "bknet.5a" not in MODEL_REGISTRY

    for model_num in range(1, 16):
        base_config = MODEL_REGISTRY[f"bknet.{model_num}"]
        thick_config = MODEL_REGISTRY[f"bknet.{model_num}t"]

        for key, value in base_config.items():
            assert thick_config[key] == value
        assert thick_config["thick_stem"] is True
        assert (
            thick_config["thick_stem_conv_ch"] == base_config["stage_out_chs"][0] // 2
        )
