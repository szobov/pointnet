import torch
from pointnet.common.model import calculate_loss, feature_regularization
from pointnet.segmentation.model import PointNet


def test_pointnet():
    batch = torch.rand(5, 3, 1000, requires_grad=True)
    tnet = PointNet(points_number=1000, number_of_classes=40)
    [scores, feature_transform_matrix] = tnet.forward(batch)
    assert scores.shape == (5, 40, 1000)
    scores.sum().backward()
    assert batch.grad.numel()


def test_feature_regularization():
    batch = torch.rand(5, 3, 1000, requires_grad=True)
    tnet = PointNet(points_number=1000, number_of_classes=40)
    [_, feature_transform_matrix] = tnet.forward(batch)
    L_reg = feature_regularization(feature_transform_matrix)
    assert L_reg.shape == torch.Size([])
    assert L_reg != torch.Tensor([0])


def test_loss():
    batch = torch.rand(5, 3, 1000, requires_grad=True)
    cls = torch.randint(0, 40, (5, 1000))
    tnet = PointNet(points_number=1000, number_of_classes=40)
    [scores, _] = tnet.forward(batch)
    loss = calculate_loss(scores, cls)
    assert loss.shape == torch.Size([])
    assert loss != torch.Tensor([0])
