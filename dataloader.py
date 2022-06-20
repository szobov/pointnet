import pathlib
import itertools

import typer
import trimesh

from torch.utils.data import Dataset


class ModelNet(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, train: bool = True):
        super().__init__()
        self._classes: dict[str, int] = {
            cls: idx for idx, cls in enumerate(sorted(map(
                lambda f: f.stem,
                filter(lambda f: f.is_dir(), dataset_dir.iterdir())
            )))
        }
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

    @property
    def files(self) -> list[pathlib.Path]:
        return self._files

    def __getitem__(self, index: int):
        file_path = self.files[index]
        cls = self.classes[file_path.parents[1].stem]
        return trimesh.load(str(file_path)), cls

    def __len__(self) -> int:
        return(len(self.files))


def __test__(dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/ModelNet40")):
    dataset = ModelNet(dataset_dir)
    assert len(dataset.classes) == 40
    assert dataset.classes["airplane"] == 0
    assert len(dataset) == 9843
    assert dataset[0][1] == 0


if __name__ == '__main__':
    typer.run(__test__)

