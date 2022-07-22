import torch
from pointnet.common.model import TNet


def test_tnet():
    batch = torch.rand(5, 3, 1000, requires_grad=True)
    tnet = TNet(feature_transfom_net=False)
    result = tnet.forward(batch)
    assert result.shape == (5, 3, 3)
    result.sum().backward()
    assert batch.grad.numel()

    tnet_feat = TNet(feature_transfom_net=True)
    batch = torch.rand(5, 64, 1000, requires_grad=True)
    result = tnet_feat.forward(batch)
    assert result.shape == (5, 64, 64)
