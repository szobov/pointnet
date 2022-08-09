import datetime
import logging
import pathlib
import typing as _t
from collections import defaultdict

import numpy as np
import rich.progress
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..common.profiler_utils import get_profiler
from ..common.train_utils import get_optimizer, process_batch
from .dataset import ShapeNet, get_data_loader
from .model import PointNet
from .train_utils import estimate_prediciton

LOGGER = logging.getLogger(__name__)


def _train_loop(dataloader: DataLoader, model: torch.nn.Module,
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR,
                device: torch.device,
                profiler: _t.Optional[torch.profiler.profile]):
    w_reg = 0.001
    assert isinstance(dataloader.dataset, ShapeNet)
    size = len(dataloader.dataset)
    model.train()
    for idx, batch in rich.progress.track(enumerate(dataloader), description="Training"):
        optimizer.zero_grad(set_to_none=True)

        (points, labels, _) = batch
        (loss, loss_regularization, scores, labels) = process_batch(points, labels, model, device)
        loss = loss + loss_regularization * w_reg

        loss.backward()
        optimizer.step()
        if profiler is not None:
            profiler.step()
        if idx % 50 == 0:
            validation_result = estimate_prediciton(scores, labels)
            current = idx * len(batch[0])
            logging.info(
                "loss: %.7f, regularization loss: %.7f, accuracy: %.1f%%, [%5d/%5d]",
                loss.item(), loss_regularization.item(),
                100 * validation_result.accuracy, current, size)
    scheduler.step()


def _test_loop(dataloader: DataLoader, model: torch.nn.Module, device: torch.device):

    assert isinstance(dataloader.dataset, ShapeNet)
    num_batches = len(dataloader)

    test_loss, accuracy = np.float64(0.0), np.float64(0.0)
    accumulated_iou_per_category: defaultdict[int, list[np.float64, int]] = defaultdict(lambda: [np.float64(0.0), 0])

    model.eval()

    with torch.no_grad():
        for batch in rich.progress.track(dataloader, description="Validating"):
            (points, labels, classes) = batch
            (loss, loss_regularization, scores, labels) = process_batch(points, labels, model, device)
            test_loss += loss.cpu().item()
            validation_result = estimate_prediciton(scores, labels)
            accuracy += validation_result.accuracy
            for category_class, iou_per_shape in zip(classes,
                                                     validation_result.average_iou_per_shape):
                accumulated_iou_per_category[category_class.item()][0] += iou_per_shape
                accumulated_iou_per_category[category_class.item()][1] += 1

    test_loss /= num_batches
    accuracy /= num_batches
    LOGGER.info(f"Test Error: Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>7f}")
    LOGGER.info("Average IoU per category:")
    for category_class, (accumulated_iou, items_number) in sorted(accumulated_iou_per_category.items(),
                                                                  key=lambda item: item[1][0]):
        averaged_iou = accumulated_iou / items_number
        LOGGER.info(f"Category: {dataloader.dataset.get_class_description(category_class)}, IoU: {(100*averaged_iou):>0.1f}%")


def train(dataset_dir: pathlib.Path,
          pretrained_model_path: _t.Optional[pathlib.Path] = None,
          use_device: _t.Optional[str] = None,
          log_dir: pathlib.Path = pathlib.Path("log/"),
          batch_size: int = 32, epoch_number: int = 250,
          dataloader_workers_num: int = 12, enable_profiler: bool = False):
    dataset_dir = dataset_dir.resolve()
    assert dataset_dir.exists()
    assert log_dir.exists()
    run_id = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

    device: torch.device
    if use_device:
        device = torch.device(use_device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    LOGGER.info("Use device: %s", device)

    train_dataloader = get_data_loader(dataset_dir, is_train=True, batch_size=batch_size,
                                       dataloader_workers_num=dataloader_workers_num, device=device)
    test_dataloader = get_data_loader(dataset_dir, is_train=True, batch_size=batch_size,
                                      dataloader_workers_num=dataloader_workers_num, device=device)

    assert isinstance(train_dataloader.dataset, ShapeNet)
    model = PointNet(train_dataloader.dataset.points_number,
                     train_dataloader.dataset.labels_number)
    if pretrained_model_path is not None and pretrained_model_path.exists():
        LOGGER.info("Use pretrained model from: %s", pretrained_model_path)
        model.load_state_dict(torch.load(str(pretrained_model_path)))

    writer = SummaryWriter(str(log_dir))
    writer.add_graph(model, torch.rand(5, 3, train_dataloader.dataset.points_number))

    model = model.to(device)

    if device == torch.device('cuda'):
        assert all(map(lambda p: p.is_cuda, model.parameters()))

    (optimizer, scheduler) = get_optimizer(model)

    for t in range(epoch_number):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        if enable_profiler:
            with get_profiler(device, log_dir, run_id) as profiler:
                _train_loop(train_dataloader, model, optimizer, scheduler, device, profiler)
        else:
            _train_loop(train_dataloader, model, optimizer, scheduler, device, profiler=None)
        _test_loop(test_dataloader, model, device)

    save_model_path = str(log_dir / f"model-{run_id}.pth")
    LOGGER.info("Training is finished, saving model: %s", save_model_path)
    torch.save(model.state_dict(), save_model_path)
