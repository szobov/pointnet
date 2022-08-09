import dataclasses
import json
import logging
import pathlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.data_processing import normalize_to_unit_sphere

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ShapeNetData:
    points_files: list[pathlib.Path] = dataclasses.field(default_factory=list)
    points_label_files: list[pathlib.Path] = dataclasses.field(default_factory=list)
    images_files: list[pathlib.Path] = dataclasses.field(default_factory=list)
    class_nums: list[int] = dataclasses.field(default_factory=list)

    def add_entity(
            self, class_num: int,
            points_file: pathlib.Path, points_label_file: pathlib.Path, image_file: pathlib.Path):
        self.points_files.append(points_file)
        self.points_label_files.append(points_label_file)
        self.images_files.append(image_file)
        self.class_nums.append(class_num)


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

        cache_file = dataset_dir / "cached_global_part_labels.json"
        LOGGER.info("Use cache file: %s", cache_file)
        need_cache = not cache_file.exists()

        class_to_part_labels_num: defaultdict[int, int] = defaultdict(lambda: 0)

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
                if need_cache:
                    max_part_label = np.max(np.loadtxt(points_label_file).astype(np.int8))
                    if class_to_part_labels_num[int(class_offset)] < max_part_label:
                        class_to_part_labels_num[int(class_offset)] = max_part_label
                image_file = dataset_dir / class_offset / "seg_img" / f"{sample_name}.png"
                assert image_file.exists()
                self._data.add_entity(int(class_offset),
                                      points_file, points_label_file, image_file)
        self._class_part_label_to_global_label: dict[tuple[int, int], int] = {}
        if need_cache:
            LOGGER.info("Cache the mapping from (class + local part label) to global part label")
            part_label_global_index = 0
            for class_num, part_labels_num in sorted(class_to_part_labels_num.items(), key=lambda i: i[0]):
                for part_label in range(1, part_labels_num + 1):
                    self._class_part_label_to_global_label[(class_num, part_label)] = part_label_global_index
                    part_label_global_index += 1
            cache_file.write_text(
                json.dumps(list(sorted(self._class_part_label_to_global_label.keys(), key=lambda i: i[0]))))
        else:
            LOGGER.info("Load from cache the mapping from (class + local part label) to global part label")
            self._class_part_label_to_global_label = {
                tuple(v): index for index, v in enumerate(json.loads(cache_file.read_text()))
            }
        self._global_label_to_class_and_local_label = {v: k for k, v in
                                                       self._class_part_label_to_global_label.items()}
        self._labels_number = len(self._class_part_label_to_global_label)
        LOGGER.debug("Number of part's labels: %s", self._labels_number)
        assert (len(self._data.images_files) ==
                len(self._data.points_label_files) ==
                len(self._data.images_files) ==
                len(self._data.class_nums))

    @property
    def labels_number(self) -> int:
        return self._labels_number

    @property
    def class_number(self) -> int:
        return len(self._classes)

    def get_class_description(self, class_number: int) -> str:
        return self._classes[class_number]

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

    def global_point_label_to_class_and_local(self, global_label: int) -> tuple[int, int]:
        return self._global_label_to_class_and_local_label[global_label]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        points = np.loadtxt(self._data.points_files[index])
        class_num = self._data.class_nums[index]
        points_label_local = np.loadtxt(self._data.points_label_files[index]).astype(np.int8)
        points_label = np.array([self._class_part_label_to_global_label[(class_num, i)]
                                 for i in points_label_local])
        points = normalize_to_unit_sphere(points)

        if len(points) < self._points_number:
            points, points_label = self._upsample_point_cloud(points,
                                                              points_label)

        shuffled_indexes = np.random.choice(
            self._points_number, self._points_number, replace=True)

        points = points[shuffled_indexes]
        points_label = points_label[shuffled_indexes]

        points_tensor = torch.from_numpy(points.T.astype(np.float32))
        points_label_tensor = torch.from_numpy(points_label)

        return points_tensor, points_label_tensor, class_num

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


def test_shapenet(dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/shapenetcore_partanno_segmentation_benchmark_v0/")):
    dataset = ShapeNet(dataset_dir)
    item = dataset[0]
    points, labels, cls = item
    assert len(points[0]) == len(labels), "Number of labels should be equal to number of points"
    assert labels.shape == (dataset.points_number, )
    assert points.shape == (3, dataset.points_number)
    assert labels[0] >= 0
    assert dataset.labels_number == 50

    dataloader = get_data_loader(dataset_dir, is_train=True, batch_size=16,
                                 dataloader_workers_num=4, device=torch.device("cpu"))
    for batch in dataloader:
        (_, batch_labels, _) = batch
        assert all(map(lambda labels: min(labels.numpy()) >= 0 and max(labels.numpy()) < 50,
                       batch_labels))
        return
