import torch

from britekit.models.mobilenet import MODEL_REGISTRY, _gen_mobilenet


def test_mobilenet_family():
    expected_parameters = {
        "mobilenet.1": 970_584,
        "mobilenet.2": 1_984_376,
        "mobilenet.3": 2_965_704,
        "mobilenet.4": 4_058_328,
        "mobilenet.5": 5_002_336,
        "mobilenet.6": 5_923_672,
    }

    for model_type, parameter_count in expected_parameters.items():
        model = _gen_mobilenet(
            MODEL_REGISTRY[model_type],
            in_chans=1,
            global_pool="",
            num_classes=0,
        )
        attention = [
            block
            for stage in model.blocks
            for block in stage
            if block.__class__.__name__ == "MobileAttention"
        ]
        assert len(attention) == 2
        assert sum(p.numel() for p in model.parameters()) == parameter_count


def test_mobilenet_feature_shape():
    model = _gen_mobilenet(
        MODEL_REGISTRY["mobilenet.4"],
        in_chans=1,
        global_pool="",
        num_classes=0,
    )

    with torch.no_grad():
        output = model(torch.zeros(1, 1, 128, 256))

    assert output.shape == (1, 1280, 4, 8)
