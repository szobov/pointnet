import pathlib
import typing as _t
import logging
import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import rich.progress

from .dataset import get_data_loader, ModelNet
from .train_utils import get_optimizer, process_batch, estimate_prediciton
from .model import PointNet
from .profiler_utils import get_profiler


LOGGER = logging.getLogger(__name__)


def _train_loop(dataloader: DataLoader, model: torch.nn.Module,
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR,
                device: torch.device,
                profiler: _t.Optional[torch.profiler.profile]):
    w_reg = 0.001
    assert isinstance(dataloader.dataset, ModelNet)
    size = len(dataloader.dataset)
    model.train()
    for idx, batch in rich.progress.track(enumerate(dataloader), description="Training"):
        optimizer.zero_grad(set_to_none=True)

        (loss, loss_regularization, scores, classes) = process_batch(batch, model, device)
        loss = loss + loss_regularization * w_reg

        loss.backward()
        optimizer.step()
        if profiler is not None:
            profiler.step()
        if idx % 50 == 0:
            correct = estimate_prediciton(scores, classes) / len(classes)
            current = idx * len(batch[0])
            logging.info(
                "loss: %.7f, regularization loss: %.7f, correct: %.1f%, [%5d/%5d]",
                loss.item(), loss_regularization.item(), 100 * correct, current, size)
    scheduler.step()


def _test_loop(dataloader: DataLoader, model: torch.nn.Module, device: torch.device):

    assert isinstance(dataloader.dataset, ModelNet)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = np.float64(0.0), np.float64(0.0)

    model.eval()

    with torch.no_grad():
        for batch in rich.progress.track(dataloader, description="Validating"):
            (loss, _, scores, classes) = process_batch(batch, model, device)
            test_loss += loss.cpu().item()
            correct += estimate_prediciton(scores, classes)

    test_loss /= num_batches
    correct /= size
    LOGGER.info(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>7f} \n")


def train(dataset_dir: pathlib.Path,
          pretrained_model_path: _t.Optional[pathlib.Path] = None,
          use_device: _t.Optional[str] = None,
          log_dir: pathlib.Path = pathlib.Path("log/"),
          batch_size: int = 32, epoch_number: int = 250,
          dataloader_workers_num: int = 12, enable_profiler: bool = False):

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

    assert isinstance(train_dataloader.dataset, ModelNet)
    model = PointNet(len(train_dataloader.dataset.classes))
    if pretrained_model_path is not None and pretrained_model_path.exists():
        LOGGER.info("Use pretrained model from: %s", pretrained_model_path)
        model.load_state_dict(torch.load(str(pretrained_model_path)))

    writer = SummaryWriter(str(log_dir))
    writer.add_graph(model, torch.rand(5, 3, train_dataloader.dataset._points_number))

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
