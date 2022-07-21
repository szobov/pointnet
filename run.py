import logging

import typer

from pointnet.classification import train as classification_train
from pointnet.classification.dataset import test_modelnet
from pointnet.classification.visualization_utils import visualize_modelnet
from pointnet.segmentation.dataset import test_shapenet
from pointnet.segmentation.visualization_utils import visualize_shapenet

entry_point = typer.Typer()


entry_point.command()(test_modelnet)
entry_point.command()(test_shapenet)
entry_point.command()(visualize_modelnet)
entry_point.command()(visualize_shapenet)
entry_point.command(name="train-classification")(classification_train.train)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    entry_point()
