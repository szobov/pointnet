import pathlib
import itertools

import typer
import trimesh

import numpy as np

import torch
from torch.utils.data import Dataset


class ModelNet(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, train: bool = True,
                 points_number: int = 1000):
        super().__init__()
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
        sampled_points = mesh.sample(count=self._points_number)
        points = torch.from_numpy(sampled_points.astype(np.float32).T)
        return points, torch.from_numpy(np.array([cls]).astype(np.int64))

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

