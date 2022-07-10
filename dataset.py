import pathlib
import itertools

import typer
import trimesh

import numpy as np

import torch
from torch.utils.data import Dataset


class ModelNet(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, train: bool = True,
                 points_number: int = 1024):
        super().__init__()
        self._train = train
        self._points_number = points_number
        self._classes: dict[str, int] = {
            cls: idx for idx, cls in enumerate(sorted(map(
                lambda f: f.stem,
                filter(lambda f: f.is_dir(), dataset_dir.iterdir())
            )))
        }
        self._classes_strings = {v: k for k, v in self._classes.items()}
        dataset_type = "train" if train else "test"
        self._files: list[pathlib.Path] = list(
            filter(lambda f: f.is_file(),
                   itertools.chain(
                       *map(lambda dir: list(dir.iterdir()),
                            map(lambda cls: dataset_dir / cls / dataset_type,
                                self.classes.keys())
                            )
                   ))
        )

    @property
    def classes(self) -> dict[str, int]:
        return self._classes

    def get_class_description(self, class_num: int) -> str:
        return self._classes_strings[class_num]

    @property
    def files(self) -> list[pathlib.Path]:
        return self._files

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[index]
        cls = self.classes[file_path.parents[1].stem]
        mesh = trimesh.load(str(file_path))
        sampled_points = mesh.sample(count=self._points_number).astype(np.float32)
        if self._train:
            sampled_points = self._augment_points(sampled_points)

        points = torch.from_numpy(sampled_points.T)
        return points, torch.from_numpy(np.array([cls]).astype(np.int64))

    def _augment_points(self, points: np.ndarray) -> np.ndarray:
        return self._jitter_points(
                self._rotate_around_z_axis(
                    self._normalize_to_unit_sphere(
                        self._shuffle_points(points)
                    )
                )
            )

    def _shuffle_points(self, points: np.ndarray) -> np.ndarray:
        return points[np.random.choice(len(points), len(points), replace=True), :]

    def _normalize_to_unit_sphere(self, points: np.ndarray) -> np.ndarray:
        # I didn't get what they meant by "normalize into a unit sphere",
        # but this normalization should feat this definition
        points /= np.max(np.linalg.norm(points - np.mean(points), axis=-1))
        assert len(points) == self._points_number
        return points

    def _rotate_around_z_axis(self, points: np.ndarray) -> np.ndarray:
        # In the paper they mentioned "along up-axis", but the problem that
        # "up-axis" depends on how you chose the order. I assume they
        # used right-hand rule and "up-axis" will be "z-axis". I also
        # assumed that "along" in this case means "around".
        angle = 2 * np.pi * np.random.uniform()
        cos = np.cos(angle)
        sin = np.sin(angle)
        # https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
        rotation_matrix = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])
        rotated_points = np.dot(rotation_matrix, points.T).T
        assert points.shape == rotated_points.shape
        return rotated_points

    def _jitter_points(self, points: np.ndarray) -> np.ndarray:
        jitter = 0.02 * np.random.randn(*points.shape)
        assert jitter.shape == points.shape
        return points + jitter

    def __len__(self) -> int:
        return(len(self.files))


def __test__(dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/ModelNet40")):
    dataset = ModelNet(dataset_dir)
    assert len(dataset.classes) == 40
    assert dataset.classes["airplane"] == 0
    assert len(dataset) == 9843
    item = dataset[0]
    assert item[1].data == torch.tensor([0])
    assert item[0].shape[0] == 3, f"{item[0][0].shape} = (3, n)"


if __name__ == '__main__':
    typer.run(__test__)

