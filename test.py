import torch

from model import TNet


def test_tnet():
    batch = torch.rand(5, 3, 1000, requires_grad=True)
    tnet = TNet(feature_transfom_net=False)
    result = tnet.forward(batch)
    assert result.shape == (5, 3, 3)
    result.sum().backward()
    assert batch.grad.numel()

    tnet_feat = TNet(feature_transfom_net=True)
    result = tnet_feat.forward(batch)
    assert result.shape == (5, 64, 64)


