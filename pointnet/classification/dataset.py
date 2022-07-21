import itertools
import logging
import pathlib

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from ..common.data_processing import normalize_to_unit_sphere

LOGGER = logging.getLogger(__name__)


class ModelNet(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, train: bool = True,
                 points_number: int = 1024, use_cache: bool = True):
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
        self._use_cache = use_cache
        if self._use_cache:
            self._cached_files: list[pathlib.Path] = []
            self._build_cache(dataset_dir)

    def _build_cache(self, dataset_dir: pathlib.Path):
        cache_dir = dataset_dir.parent / f"cached_{dataset_dir.name}"
        for file_path in self.files:
            cache_file_path = file_path.relative_to(dataset_dir).with_suffix(".npy")
            cache_file_path = cache_dir / cache_file_path
            if cache_file_path.exists():
                cached_points_number = len(np.load(str(cache_file_path)))
                if self._points_number == cached_points_number:
                    self._cached_files.append(cache_file_path)
                    continue
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            sampled_points = self._load_mesh_and_sample_points(file_path,
                                                               self._points_number)
            LOGGER.info("Save cached file in: %s", cache_file_path)
            np.save(cache_file_path, sampled_points)
            self._cached_files.append(cache_file_path)
        assert len(self._cached_files) == len(self._files)

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
        if self._use_cache:
            sampled_points = np.load(self._cached_files[index])
        else:
            sampled_points = self._load_mesh_and_sample_points(file_path,
                                                               self._points_number)
        normalized_points = normalize_to_unit_sphere(sampled_points)
        assert len(normalized_points) == self._points_number
        if self._train:
            augmented_points = self._augment_points(normalized_points)
            assert augmented_points.shape == sampled_points.shape
            sampled_points = augmented_points

        points = torch.from_numpy(sampled_points.T.astype(np.float32))
        return points, torch.from_numpy(np.array(cls).astype(np.int64))

    def _load_mesh_and_sample_points(self, file_path: pathlib.Path,
                                     points_number: int) -> np.ndarray:
        mesh = trimesh.load(str(file_path))
        return mesh.sample(count=points_number)

    def _augment_points(self, points: np.ndarray) -> np.ndarray:
        return self._jitter_points(
                self._rotate_around_z_axis(
                    self._shuffle_points(points)
                )
            )

    def _shuffle_points(self, points: np.ndarray) -> np.ndarray:
        return points[np.random.choice(len(points), len(points), replace=True), :]

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

    @property
    def points_number(self) -> int:
        return self._points_number

    def __len__(self) -> int:
        return(len(self.files))


def get_data_loader(path_to_dataset: pathlib.Path, is_train: bool,
                    batch_size: int, dataloader_workers_num: int,
                    device: torch.device) -> torch.utils.data.DataLoader:
    dataset = ModelNet(path_to_dataset, train=is_train)
    pin_memory = False
    if device == torch.device('cuda'):
        pin_memory = True
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory,
        shuffle=True, num_workers=dataloader_workers_num)


def test_modelnet(dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/ModelNet40")):
    dataset = ModelNet(dataset_dir)
    assert len(dataset.classes) == 40
    assert dataset.classes["airplane"] == 0
    assert len(dataset) == 9843
    item = dataset[0]
    assert item[1].data == torch.tensor([0])
    assert item[0].shape[0] == 3, f"{item[0][0].shape} = (3, n)"
