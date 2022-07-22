import torch
from torch import nn

# jit_nn_module = torch.jit.ScriptModule
# jit_script_method = torch.jit.script_method
# jit_script = torch.jit.script
jit_nn_module = nn.Module
jit_script_method = lambda f: f
jit_script = lambda f: f


class GlobalMaxPooling(jit_nn_module):
    @jit_script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, 2)[0]


class TNet(jit_nn_module):

    def __init__(self, feature_transfom_net: bool):
        super().__init__()
        # output 3x3 matrix (initialized as identity matrix)
        # TODO: move from nn.Sequential, since it's very difficult to debug it
        self._is_feature_transform_net = feature_transfom_net
        self._expected_input_dim = 3
        if self._is_feature_transform_net:
            self._expected_input_dim = 64

        self._expected_output_dim = self._expected_input_dim * self._expected_input_dim
        self._output_matrix_shape = (self._expected_input_dim, self._expected_input_dim)

        self._layers: list[nn.Module] = [
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
        ]

        # The didn't clearly mention it in paper
        self._layers.append(nn.Linear(256, self._expected_output_dim))

        self._model = nn.Sequential(*self._layers)

        self._initial_matrix = torch.nn.init.eye_(
            torch.empty(size=self._output_matrix_shape)).reshape(1, self._expected_output_dim)

    @jit_script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self._expected_input_dim, f"{x.shape}, {self._expected_input_dim}"
        result = self._model(x)

        assert result.shape[-1] == self._expected_output_dim, f"{result.shape[-1]} != {self._expected_output_dim}"

        initial_matrix = self._initial_matrix.repeat(x.shape[0], 1).to(x.device)

        assert initial_matrix.shape == result.shape, f"{initial_matrix.shape} != {result.shape}"
        new_shape = result.shape[:-1] + self._output_matrix_shape
        return (initial_matrix + result).reshape(new_shape)


@jit_script
def feature_regularization(feature_transform_matrix: torch.Tensor):
    A_A_T = torch.bmm(feature_transform_matrix, torch.transpose(feature_transform_matrix, 1, 2))
    identity = (torch.eye(feature_transform_matrix.shape[-1]).
                repeat(feature_transform_matrix.shape[0], 1).
                reshape(A_A_T.shape)).to(feature_transform_matrix.device)
    return torch.mean(torch.linalg.matrix_norm(identity - A_A_T, ord="fro"))


@jit_script
def calculate_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(scores, target)
