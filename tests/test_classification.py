import torch
from pointnet.classification.model import PointNet
from pointnet.common.model import calculate_loss, feature_regularization


def test_pointnet():
    batch = torch.rand(5, 3, 1000, requires_grad=True)
    tnet = PointNet(number_of_classes=16)
    [scores, feature_transform_matrix] = tnet.forward(batch)
    assert scores.shape == (5, 16)
    scores.sum().backward()
    assert batch.grad.numel()


def test_feature_regularization():
    batch = torch.rand(5, 3, 1000)
    tnet = PointNet(number_of_classes=16)
    [_, feature_transform_matrix] = tnet.forward(batch)
    L_reg = feature_regularization(feature_transform_matrix)
    assert L_reg.shape == torch.Size([])
    assert L_reg != torch.Tensor([0])


def test_loss():
    batch = torch.rand(5, 3, 1000)
    cls = torch.randint(0, 16, (5, ))
    tnet = PointNet(number_of_classes=16)
    [scores, _] = tnet.forward(batch)
    loss = calculate_loss(scores, cls)
    assert loss.shape == torch.Size([])
    assert loss != torch.Tensor([0])
