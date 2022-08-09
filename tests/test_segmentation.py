import numpy as np
import torch
from pointnet.common.model import calculate_loss, feature_regularization
from pointnet.segmentation.model import PointNet
from pointnet.segmentation.train_utils import estimate_prediciton


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


def test_estimate_prediction_smoke():
    scores = torch.randn(5, 40, 1000)
    cls = torch.randint(0, 40, (5, 1000))
    estimation_result = estimate_prediciton(scores, cls)
    assert estimation_result.accuracy >= 0.0
    assert len(estimation_result.average_iou_per_shape) == 5
    assert all(estimation_result.average_iou_per_shape)


def test_estimate_prediction_all_same():
    scores = torch.zeros((2, 5, 1000))
    expected_class = 2
    scores[:, expected_class, :] = 1
    expected_label_per_points = torch.ones((2, 1000)).type(torch.int) * expected_class
    estimation_result = estimate_prediciton(scores, expected_label_per_points)
    assert estimation_result.accuracy == 1.0
    assert np.allclose(estimation_result.average_iou_per_shape, [1.0, 1.0])


def test_estimate_prediction_half_same():
    scores = torch.zeros((2, 5, 1000))
    expected_class = 2
    scores[:, expected_class, :] = 1
    expected_label_per_points = torch.ones((2, 1000)).type(torch.int) * expected_class
    expected_label_per_points[1, :] = 0
    estimation_result = estimate_prediciton(scores, expected_label_per_points)
    assert estimation_result.accuracy == .5
    assert np.allclose(estimation_result.average_iou_per_shape, [1.0, .0])


def test_estimate_prediction_wrong_prediction():
    scores = torch.zeros((2, 5, 1000))
    expected_class = 2
    scores[:, expected_class, :] = 1
    expected_label_per_points = torch.ones((2, 1000)).type(torch.int)
    expected_label_per_points[1, :] = 0
    estimation_result = estimate_prediciton(scores, expected_label_per_points)
    assert estimation_result.accuracy == .0
    assert np.allclose(estimation_result.average_iou_per_shape, [.0, .0])
