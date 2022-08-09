import dataclasses

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class BatchValidationResult:
    accuracy: float
    average_iou_per_shape: np.ndarray


def estimate_prediciton(score_per_points: torch.Tensor, expected_label_per_point: torch.Tensor) -> BatchValidationResult:
    predicted_label_per_point = score_per_points.argmax(1)
    assert predicted_label_per_point.shape == expected_label_per_point.shape, \
        f"{predicted_label_per_point.shape} == {expected_label_per_point.shape}"
    predicted_correctly = predicted_label_per_point == expected_label_per_point
    accuracy = predicted_correctly.type(torch.float).sum().cpu().item() / expected_label_per_point.numel()
    average_iou_per_shape = np.zeros(score_per_points.shape[0])

    for index, (pred, expected) in enumerate(zip(predicted_label_per_point,
                                                 expected_label_per_point)):
        expected_labels = set(expected.cpu().numpy())
        per_shape_predicted_correctly = pred == expected
        for part_label in expected_labels:
            per_part_prediction = pred == part_label
            per_part_expected = expected == part_label
            per_part_intersection = (per_part_expected * per_shape_predicted_correctly).sum()
            per_part_union = per_part_prediction.sum() + per_part_expected.sum() - per_part_intersection
            per_part_iou = 1.0
            if per_part_union != 0:
                per_part_iou = (per_part_intersection / per_part_union).cpu().item()
            average_iou_per_shape[index] += per_part_iou
        average_iou_per_shape[index] /= len(expected_labels)
    return BatchValidationResult(accuracy, average_iou_per_shape)



