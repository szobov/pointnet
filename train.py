import pathlib
import typing as _t

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import rich.progress

from dataset import ModelNet
from model import PointNet, feature_regularization, calculate_loss


def get_optimizer(model: torch.nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    learning_rate = 0.001
    learning_rate_decay_step = 20
    learning_rate_decay = 0.5
    # they mentioned it in the paper, but Adam optimizer
    # doesn't have such a parameter
    # momentum = 0.9
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=learning_rate_decay_step,
                                                gamma=learning_rate_decay)
    return optimizer, scheduler


def train_loop(dataloader: DataLoader, model: torch.nn.Module,
               optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR,
               device: torch.device,
               profiler: _t.Optional[torch.profiler.profile]):
    w_reg = 0.001

    size = len(dataloader.dataset)
    model.train()
    for idx, batch in rich.progress.track(enumerate(dataloader), description="Training"):
        (points, cls) = batch
        (points, cls) = points.to(device, non_blocking=True), cls.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        (scores, feature_transform_matrix) = model(points)
        L_reg = feature_regularization(feature_transform_matrix)
        loss = calculate_loss(scores, cls)
        loss = loss + L_reg * w_reg

        loss.backward()
        optimizer.step()
        if profiler is not None:
            profiler.step()
        if idx % 50 == 0:
            correct = (scores.argmax(1) == cls).type(torch.float).sum().cpu().item()
            current = idx * len(points)
            print(f"loss: {loss.item():>7f}, regularization loss: {L_reg.item():>7f}, correct: {correct:>7f},  [{current:>5d}/{size:>5d}]")
    scheduler.step()


def test_loop(dataloader: DataLoader, model: torch.nn.Module, device: torch.device):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = np.float64(0.0), np.float64(0.0)

    model.eval()

    with torch.no_grad():
        for batch in rich.progress.track(dataloader, description="Validating"):
            (points, cls) = batch
            (points, cls) = points.to(device, non_blocking=True), cls.to(device, non_blocking=True)
            (scores, _) = model(points)
            loss = calculate_loss(scores, cls)
            test_loss += loss.cpu()
            predicted_classes = scores.argmax(1)
            assert predicted_classes.shape == cls.shape, f"{predicted_classes.shape} == {cls.shape}"
            correct += (predicted_classes == cls).type(torch.float).sum().cpu().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>7f} \n")


def main():
    dataset_dir = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/ModelNet40")
    batch_size = 32
    epoch_number = 250
    dataloader_workers_num = 12
    enable_profiler = False

    device: torch.device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Use device: ", device)

    writer = SummaryWriter('log/profile')
    train_dataset = ModelNet(dataset_dir, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,
                                  shuffle=True, num_workers=dataloader_workers_num)

    test_dataset = ModelNet(dataset_dir, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                                 shuffle=True, num_workers=dataloader_workers_num)

    model = PointNet(len(train_dataset.classes))
    writer.add_graph(model, torch.rand(5, 3, 1000))
    model = model.to(device)
    if device == torch.device('cuda'):
        assert all(map(lambda p: p.is_cuda, model.parameters()))

    (optimizer, scheduler) = get_optimizer(model)

    for t in range(epoch_number):
        print(f"Epoch {t+1}\n-------------------------------")
        if enable_profiler:
            with torch.profiler.profile(
                    activities=[
                                torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(
                        wait=2,
                        warmup=2,
                        active=6,
                        repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('log/profile'),
                    profile_memory=True,
                    record_shapes=True,
                    with_flops=True,
                    with_stack=True
            ) as profiler:
                train_loop(train_dataloader, model, optimizer, scheduler, device, profiler)
        else:
            train_loop(train_dataloader, model, optimizer, scheduler, device, profiler=None)
        test_loop(test_dataloader, model, device)
    torch.save(model.state_dict(), "log/model.pth")

if __name__ == '__main__':
    main()
