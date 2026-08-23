import timm
import torch
from timm.models import repvit

from britekit.models.repvit import MODEL_REGISTRY


def test_repvit_parameter_targets():
    expected = {
        "repvit.1": 1_039_116,
        "repvit.2": 1_973_288,
        "repvit.3": 3_065_172,
        "repvit.4": 3_963_252,
        "repvit.5": 4_717_360,
        "repvit.6": 5_039_008,
        "repvit.7": 6_003_952,
    }

    for model_type, parameter_count in expected.items():
        embed_dim, depth = MODEL_REGISTRY[model_type]
        model = repvit.RepVit(
            embed_dim=embed_dim,
            depth=depth,
            in_chans=1,
            global_pool="",
            num_classes=0,
        )
        assert sum(p.numel() for p in model.parameters()) == parameter_count


def test_repvit_5_matches_timm_m0_9():
    embed_dim, depth = MODEL_REGISTRY["repvit.5"]
    model = repvit.RepVit(
        embed_dim=embed_dim,
        depth=depth,
        in_chans=1,
        global_pool="",
        num_classes=0,
    )
    timm_model = timm.create_model(
        "repvit_m0_9",
        in_chans=1,
        global_pool="",
        num_classes=0,
    )

    assert model.state_dict().keys() == timm_model.state_dict().keys()
    assert all(
        parameter.shape == timm_model.state_dict()[name].shape
        for name, parameter in model.state_dict().items()
    )


def test_repvit_feature_shape():
    embed_dim, depth = MODEL_REGISTRY["repvit.5"]
    model = repvit.RepVit(
        embed_dim=embed_dim,
        depth=depth,
        in_chans=1,
        global_pool="",
        num_classes=0,
    )

    with torch.no_grad():
        output = model(torch.zeros(1, 1, 128, 256))

    assert output.shape == (1, 384, 4, 8)
