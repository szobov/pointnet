import hashlib
import pathlib
import random
import webbrowser

import numpy as np
import scenepic as sp

from .dataset import ShapeNet
from .model import PointNet


def _generate_color(key: int) -> np.ndarray:
    return np.array(
        list(map(int, hashlib.md5(bytes(key)).digest()[:3])),
        dtype=np.float64) / 255.0


def visualize_shapenet(
        dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/shapenetcore_partanno_segmentation_benchmark_v0/"),
        output_file: pathlib.Path = pathlib.Path("/tmp/shapenet.html")
):
    dataset = ShapeNet(dataset_dir)
    scene = sp.Scene()
    segmented_points_canvas = scene.create_canvas_3d(
        "segmented points", width=400, height=400,
        shading=sp.Shading(bg_color=sp.Colors.White))
    image_canvas = scene.create_canvas_2d(
        "image", width=400, height=400)
    scene.link_canvas_events(segmented_points_canvas, image_canvas)

    for indx in random.choices(range(len(dataset)), k=10):
        (segmented_points, labels, _) = dataset[indx]
        segmented_points_frame = segmented_points_canvas.create_frame()

        image_frame = image_canvas.create_frame()
        image = scene.create_image()
        image.load(str(dataset._data.images_files[indx]))
        image_frame.add_image(image)

        points = scene.create_mesh()
        points.add_sphere(color=sp.Colors.White)
        points.apply_transform(sp.Transforms.Scale(0.01))
        points.enable_instancing(
            segmented_points.numpy().T,
            colors=np.array(list(
                map(_generate_color,
                    map(lambda l: dataset.global_point_label_to_class_and_local(int(l))[1],
                        labels)))))

        segmented_points_frame.add_mesh(points)

    scene.save_as_html(str(output_file), title="ShapeNet segmented")
    webbrowser.open(str(output_file))


def visualize_prediction(
        dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/shapenetcore_partanno_segmentation_benchmark_v0/"),
        pretrained_model: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/pretrained_models/model-10-08-2022-18:56:36.pth"),
        output_file: pathlib.Path = pathlib.Path("/tmp/shapenet.html")
):
    dataset = ShapeNet(dataset_dir, train=False)
    scene = sp.Scene()
    segmented_points_canvas = scene.create_canvas_3d(
        "segmented points", width=400, height=400,
        shading=sp.Shading(bg_color=sp.Colors.White))
    predicted_segmented_points_canvas = scene.create_canvas_3d(
        "predicted segmented points", width=400, height=400,
        shading=sp.Shading(bg_color=sp.Colors.White))
    image_canvas = scene.create_canvas_2d(
        "image", width=400, height=400)
    scene.link_canvas_events(segmented_points_canvas,
                             predicted_segmented_points_canvas,
                             image_canvas)

    model = PointNet(dataset.points_number, dataset.labels_number)
    model.eval()

    for indx in random.choices(range(len(dataset)), k=10):
        (segmented_points, labels, _) = dataset[indx]
        segmented_points_frame = segmented_points_canvas.create_frame()

        image_frame = image_canvas.create_frame()
        image = scene.create_image()
        image.load(str(dataset._data.images_files[indx]))
        image_frame.add_image(image)

        points = scene.create_mesh()
        points.add_sphere(color=sp.Colors.White)
        points.apply_transform(sp.Transforms.Scale(0.01))
        points.enable_instancing(
            segmented_points.numpy().T,
            colors=np.array(list(
                map(_generate_color,
                    map(lambda l: dataset.global_point_label_to_class_and_local(int(l))[1],
                        labels)))))

        segmented_points_frame.add_mesh(points)

    scene.save_as_html(str(output_file), title="ShapeNet segmented")
    webbrowser.open(str(output_file))
