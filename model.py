import torch
from torch import nn


class GlobalMaxPooling(nn.Module):
    def forward(self, x):
        return torch.max(x, 2)[0]


class TNet(nn.Module):

    def __init__(self, feature_transfom_net: bool):
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
        self._is_feature_transform_net = feature_transfom_net
        self._expected_input_dim = 3
        if self._is_feature_transform_net:
            self._expected_input_dim = 64

        self._expected_output_dim = self._expected_input_dim * self._expected_input_dim
        self._output_matrix_shape = (self._expected_input_dim, self._expected_input_dim)

        self._layers = [
            # I use Conv1d instead of Linear because BatchNorm1d
            # expects the input shape to be (N, C, L), but with Linear
            # it will be (N, L, C)
            nn.Conv1d(self._expected_input_dim, 64, 1),
            nn.ReLU(),
            # XXX: rewrite to use nn.functional.batch_norm since we need a custom
            #      momentum (decay) calculation, changing from 0.5 to 0.1
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
        ]

        # The didn't clearly mention it in paper
        self._layers.append(nn.Linear(256, self._expected_output_dim))

        self._model = nn.Sequential(*self._layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self._expected_input_dim, x.shape
        result = self._model(x)

        assert result.shape[-1] == self._expected_output_dim, f"{result.shape[-1]} != {self._expected_output_dim}"

        initial_matrix = torch.nn.init.eye_(
            torch.empty(size=self._output_matrix_shape)).reshape(1, self._expected_output_dim).repeat(x.shape[0], 1)

        assert initial_matrix.shape == result.shape, f"{initial_matrix.shape} != {result.shape}"
        new_shape = result.shape[:-1] + self._output_matrix_shape
        return (initial_matrix + result).reshape(new_shape)


class PointNet(nn.Module):

    def __init__(self, number_of_classes: int):
        super().__init__()
        self._number_of_classes = number_of_classes
        self._transform_net = TNet(feature_transfom_net=False)
        self._mlps_1 = nn.Sequential(*[
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ])
        self._feature_transform_net = TNet(feature_transfom_net=True)

        self._mlps_2 = nn.Sequential(*[
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        ])
        self._global_max_pool = GlobalMaxPooling()  # (batch_size, 1024)
        self._mlps_3 = nn.Sequential(*[
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        ])

        self._dropout = nn.Dropout(p=0.7)

        self._mlps_4 = nn.Sequential(*[
            nn.Linear(256, number_of_classes),
            nn.BatchNorm1d(number_of_classes),
            nn.ReLU()
        ])

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = points.shape[0]
        assert points.shape[1] == 3, points.shape

        transform_matrix = self._transform_net(points)
        assert transform_matrix.shape == (batch_size, 3, 3), transform_matrix.shape
        transformed_points = torch.bmm(transform_matrix, points)

        features = self._mlps_1(transformed_points)
        assert features.shape[1] == 64

        feature_transform_matrix = self._feature_transform_net(features)
        assert feature_transform_matrix.shape == (batch_size, 64, 64), feature_transform_matrix.shape
        transformed_features = torch.bmm(feature_transform_matrix, features)

        transformed_features = self._mlps_2(transformed_features)
        assert transformed_features.shape[1] == 1024, transformed_features.shape

        global_features = self._global_max_pool(transformed_features)
        assert global_features.shape == (batch_size, 1024), global_features.shape

        global_features = self._mlps_3(global_features)

        # it should work only if model.train() was called
        global_features = self._dropout(global_features)

        scores = self._mlps_4(global_features)
        assert scores.shape == (batch_size, self._number_of_classes), scores.shape

        return scores, feature_transform_matrix  # to calculate regularization

    @staticmethod
    def regularization(feature_transform_matrix: torch.Tensor):
        A_A_T = torch.bmm(feature_transform_matrix, torch.transpose(feature_transform_matrix, 1, 2))
        identity = (torch.eye(feature_transform_matrix.shape[-1]).
                    repeat(feature_transform_matrix.shape[0], 1).
                    reshape(A_A_T.shape))
        return torch.mean(torch.linalg.matrix_norm(identity - A_A_T, ord="fro"))

    @staticmethod
    def loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(scores, target)
