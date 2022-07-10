import torch

from model import TNet, PointNet, feature_regularization, calculate_loss


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
