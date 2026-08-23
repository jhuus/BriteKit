import torch
from timm.layers import ScaledStdConv2d, ScaledStdConv2dSame
from timm.models import nfnet

from britekit.models.export_util import fold_scaled_std_convs
from britekit.models.nfnet import MODEL_REGISTRY


def test_fold_scaled_std_convs_preserves_nfnet_output():
    model = nfnet.NormFreeNet(
        cfg=MODEL_REGISTRY["nfnet.1"],
        in_chans=1,
        num_classes=3,
    ).eval()
    input = torch.randn(2, 1, 64, 128)

    with torch.no_grad():
        expected = model(input)
        fold_scaled_std_convs(model)
        actual = model(input)

    scaled_conv_types = (ScaledStdConv2d, ScaledStdConv2dSame)
    assert not any(isinstance(module, scaled_conv_types) for module in model.modules())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
