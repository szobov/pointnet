import torch

from model import TNet, PointNet


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


def test_pointnet():
    batch = torch.rand(5, 3, 1000, requires_grad=True)
    tnet = PointNet(number_of_classes=16)
    [scores, feature_transform_matrix] = tnet.forward(batch)
    assert scores.shape == (5, 16)
    scores.sum().backward()
    assert batch.grad.numel()

