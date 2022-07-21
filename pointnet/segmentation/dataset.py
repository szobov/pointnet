import dataclasses
import json
import logging
import pathlib
import typing as _t

import numpy as np
import torch
import typer
from torch.utils.data import Dataset

from ..common.data_processing import normalize_to_unit_sphere

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ShapeNetData:
    points_files: list[pathlib.Path] = dataclasses.field(default_factory=list)
    points_label_files: list[pathlib.Path] = dataclasses.field(default_factory=list)
    images_files: list[pathlib.Path] = dataclasses.field(default_factory=list)
    classes: list[str] = dataclasses.field(default_factory=list)

    def add_entity(
            self, class_description: str,
            points_file: pathlib.Path, points_label_file: pathlib.Path, image_file: pathlib.Path):
        self.points_files.append(points_file)
        self.points_label_files.append(points_label_file)
        self.images_files.append(image_file)
        self.classes.append(class_description)


class ShapeNet(Dataset):

    _TEST_TYPES = ("test", "val")
    _TRAIN_TYPES: tuple[str, ...] = ("train", )

    def __init__(self, dataset_dir: pathlib.Path,
                 points_number: int = 2048, train: bool = True):
        super().__init__()
        self._train = train
        self._points_number = points_number
        categories_file = dataset_dir / "synsetoffset2category.txt"
        self._classes: dict[int, str] = {
            int(r[1]): r[0].lower()
            for r in map(str.split,
                         categories_file.read_text().splitlines())
        }
        use_types: tuple[str, ...] = self._TEST_TYPES
        if train:
            use_types = self._TRAIN_TYPES

        self._data = ShapeNetData()

        for file_type in use_types:
            file_name = f"shuffled_{file_type}_file_list.json"
            split_file = dataset_dir / "train_test_split" / file_name
            files = map(pathlib.Path, json.loads(split_file.read_text()))
            for path_file in files:
                class_offset = path_file.parent.name
                sample_name = path_file.name
                points_file = dataset_dir / class_offset / "points" / f"{sample_name}.pts"
                assert points_file.exists()
                points_label_file = dataset_dir / class_offset / "points_label" / f"{sample_name}.seg"
                assert points_label_file.exists()
                image_file = dataset_dir / class_offset / "seg_img" / f"{sample_name}.png"
                assert image_file.exists()
                self._data.add_entity(self._classes[int(class_offset)],
                                      points_file, points_label_file, image_file)
        assert (len(self._data.images_files) ==
                len(self._data.points_label_files) ==
                len(self._data.images_files) ==
                len(self._data.classes))

    def _upsample_point_cloud(self, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # In the paper they didn't mention how they deal with different dimentionality
        # of the samples from the dateset. I decided to, let's say, "upsample" point cloud in
        # case if the number of samples is less, then required, by sampling some points and
        # adding a small nudge and then extend the original points with it.
        difference = self._points_number - len(points)
        assert difference > 0
        samples_indexes = np.random.choice(
            len(points), difference, replace=True)
        samples = points[samples_indexes]
        points = np.append(points,
                           samples + 0.005 * np.random.randn(*samples.shape),
                           axis=0)
        labels = np.append(labels, labels[samples_indexes], axis=0)
        return points, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        points = np.array(list(
            map(lambda line: list(map(np.float64, line)),
                map(str.split,
                self._data.points_files[index].read_text().splitlines())
                ))
            )
        points_label = np.array(list(
            map(int,
                self._data.points_label_files[index].read_text().splitlines()
                )
            )
        )
        points = normalize_to_unit_sphere(points)

        if len(points) < self._points_number:
            points, points_label = self._upsample_point_cloud(points,
                                                              points_label)

        shuffled_indexes = np.random.choice(
            self._points_number, self._points_number, replace=True)

        points = points[shuffled_indexes]
        points_label = points_label[shuffled_indexes]

        points_tensor = torch.from_numpy(points.T.astype(np.float64))
        points_label_tensor = torch.from_numpy(points_label.astype(np.int32))

        return points_tensor, points_label_tensor

    def __len__(self) -> int:
        return(len(self._data.points_files))

    @property
    def points_number(self) -> int:
        return self._points_number


def get_data_loader(path_to_dataset: pathlib.Path, is_train: bool,
                    batch_size: int, dataloader_workers_num: int,
                    device: torch.device) -> torch.utils.data.DataLoader:
    dataset = ShapeNet(path_to_dataset, train=is_train)
    pin_memory = False
    if device == torch.device('cuda'):
        pin_memory = True
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory,
        shuffle=True, num_workers=dataloader_workers_num)


def test_dataset(dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/shapenetcore_partanno_segmentation_benchmark_v0/")):
    dataset = ShapeNet(dataset_dir)
    item = dataset[0]
    points, labels = item
    assert len(points[0]) == len(labels), "Number of labels should be equal to number of points"
    assert points.shape == (3, dataset.points_number)
    assert labels[0] >= 0

    dataloader = get_data_loader(dataset_dir, is_train=True, batch_size=16,
                                 dataloader_workers_num=4, device=torch.device("cpu"))
    for batch in dataloader:
        assert len(set(map(lambda item: item[0].shape[-1], batch))) == 1
        return


if __name__ == '__main__':
    typer.run(test_dataset)
