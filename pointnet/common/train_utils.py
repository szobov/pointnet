import torch

from ..common.model import calculate_loss, feature_regularization


def get_optimizer(model: torch.nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    learning_rate = 0.001
    learning_rate_decay_step = 20
    learning_rate_decay = 0.5
    # they mentioned it in the paper, but Adam optimizer
    # doesn't have such a parameter
    # momentum = 0.9
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=learning_rate_decay_step,
                                                gamma=learning_rate_decay)
    return optimizer, scheduler


def process_batch(
        points: torch.Tensor,
        labels: torch.Tensor,
        model: torch.nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    (points, labels) = points.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    (scores, feature_transform_matrix) = model(points)
    L_reg = feature_regularization(feature_transform_matrix)
    loss = calculate_loss(scores, labels)
    return loss, L_reg, scores, labels
