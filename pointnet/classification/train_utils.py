import torch


def estimate_prediciton(scores: torch.Tensor, expected_classes: torch.Tensor) -> float:
    predicted_classes = scores.argmax(1)
    assert predicted_classes.shape == expected_classes.shape, f"{predicted_classes.shape} == {expected_classes.shape}"
    return (predicted_classes == expected_classes).type(torch.float).sum().cpu().item()
