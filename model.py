import torch
from torch import nn


class GlobalMaxPooling(nn.Module):
    def forward(self, x):
        return torch.max(x, 1)[0]


class TNet(nn.Module):

    def __init__(self):
        # output 3x3 matrix (initialized as identity matrix)
        # shared MLP(64, 128, 1024) on each point
        # ReLU and batch-norm
        # max-pooling across points
        # ReLU and batch-norm
        # fully-connected with size 512
        # ReLU and batch-norm
        # fully-connected with size 256
        self._layers = nn.Sequential([
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            GlobalMaxPooling(),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Linear(256, 9)  # The didn't clearly mention it in paper
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.forward(x)
        assert result.shape[-1] == 9
        new_shape = result.shape[:-1] + (3, 3)
        return result.reshape(new_shape)


class PointNet(nn.Module):
    ...
