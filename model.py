import torch
from torch import nn


class GlobalMaxPooling(nn.Module):
    def forward(self, x):
        return torch.max(x, 2)[0]


class TNet(nn.Module):

    def __init__(self):
        super().__init__()
        # output 3x3 matrix (initialized as identity matrix)
        # shared MLP(64, 128, 1024) on each point
        # ReLU and batch-norm
        # max-pooling across points
        # ReLU and batch-norm
        # fully-connected with size 512
        # ReLU and batch-norm
        # fully-connected with size 256
        # TODO: move from nn.Sequential, since it's very difficult to debug it
        self._layers = nn.Sequential(*[
            # I use Conv1d instead of Linear because BatchNorm1d
            # expects the input shape to be (N, C, L), but with Linear
            # it will be (N, L, C)
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            GlobalMaxPooling(),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 9)  # The didn't clearly mention it in paper
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self._layers(x)
        assert result.shape[-1] == 9, f"{result.shape[-1]} != 9"

        initial_matrix = torch.nn.init.eye_(
            torch.empty(3, 3)).reshape(1, 9).repeat(x.shape[0], 1)

        assert initial_matrix.shape == result.shape, f"{initial_matrix.shape} != {result.shape}"
        new_shape = result.shape[:-1] + (3, 3)
        return (initial_matrix + result).reshape(new_shape)


class PointNet(nn.Module):
    ...
