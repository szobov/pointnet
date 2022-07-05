import pathlib

import torch
from torch.utils.data import DataLoader

from dataset import ModelNet
from model import PointNet


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
               optimizer: torch.optim.Optimizer, sheduler: torch.optim.lr_scheduler.StepLR):
    w_reg = 0.001

    size = len(dataloader.dataset)
    model.train()
    print("Training...")
    for idx, batch in enumerate(dataloader):
        (points, cls) = batch
        (scores, feature_transform_matrix) = model(points)
        L_reg = PointNet.regularization(feature_transform_matrix)
        loss = PointNet.loss(scores, torch.flatten(cls))
        loss = loss + L_reg * w_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            loss, current = loss.item(), idx * len(points)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0, 0

    model.eval()

    print("Validating...")
    with torch.no_grad():
        for batch in dataloader:
            (points, cls) = batch
            (scores, _) = model(points)
            test_loss += PointNet.loss(scores, torch.flatten(cls)).item()
            correct += (scores.argmax(1) == cls).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    dataset_dir = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/ModelNet40")
    batch_size = 32
    epoch_number = 250
    dataloader_workers_num = 8

    train_dataset = ModelNet(dataset_dir, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=dataloader_workers_num)

    test_dataset = ModelNet(dataset_dir, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=dataloader_workers_num)

    model = PointNet(len(train_dataset.classes))

    (optimizer, scheduler) = get_optimizer(model)

    for t in range(epoch_number):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, scheduler)
        test_loop(test_dataloader, model)


if __name__ == '__main__':
    main()
