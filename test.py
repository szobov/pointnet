import torch

from model import TNet


def test_tnet():
    batch = torch.rand(5, 3, 1000, requires_grad=True)
    tnet = TNet()
    result = tnet.forward(batch)
    assert result.shape == (5, 3, 3)
    result.sum().backward()
    assert batch.grad.numel()
