import torch
from torch import nn

from ..common.model import (GlobalMaxPooling, TNet, jit_nn_module,
                            jit_script_method)


class PointNet(jit_nn_module):

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

        self._dropout = nn.Dropout(p=0.3)

        self._mlps_4 = nn.Sequential(*[
            nn.Linear(256, number_of_classes),
        ])

    @jit_script_method
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
