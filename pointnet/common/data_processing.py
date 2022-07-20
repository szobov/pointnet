import numpy as np


def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    # I didn't get what they meant by "normalize into a unit sphere",
    # but this normalization should feat this definition
    points /= np.max(np.linalg.norm(points - np.mean(points), axis=-1))
    return points
