import pathlib
import random
import webbrowser

import scenepic as sp
import trimesh
import typer

from .dataset import ModelNet


def main(
        dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/ModelNet40"),
        output_file: pathlib.Path = pathlib.Path("/tmp/modelnet.html")
):
    dataset = ModelNet(dataset_dir, train=False)
    scene = sp.Scene()
    mesh_canvas = scene.create_canvas_3d(
        "mesh", width=400, height=400,
        shading=sp.Shading(bg_color=sp.Colors.White))
    points_canvas = scene.create_canvas_3d(
        "points", width=400, height=400,
        shading=sp.Shading(bg_color=sp.Colors.White))
    augmented_points_canvas = scene.create_canvas_3d(
        "augmented points", width=400, height=400,
        shading=sp.Shading(bg_color=sp.Colors.White))
    scene.link_canvas_events(mesh_canvas, points_canvas, augmented_points_canvas)
    for indx in random.choices(range(len(dataset)), k=10):
        item = dataset[indx]
        text = dataset.get_class_description(int(item[1]))
        mesh_frame = mesh_canvas.create_frame()
        points_frame = points_canvas.create_frame()
        augmented_points_frame = augmented_points_canvas.create_frame()
        mesh = scene.create_mesh()
        for face in trimesh.load(dataset.files[indx]).triangles:
            mesh.add_triangle(color=sp.Colors.Red,
                              p0=face[0], p1=face[1], p2=face[2])
        mesh.apply_transform(sp.Transforms.Scale(0.01))
        mesh_frame.add_mesh(mesh)

        points = scene.create_mesh()
        points.add_sphere(color=sp.Colors.Green)
        points.apply_transform(sp.Transforms.Scale(0.01))
        points.enable_instancing(dataset._normalize_to_unit_sphere(
            item[0].numpy().T))
        points_frame.add_mesh(points)

        augmented_points = scene.create_mesh()
        augmented_points.add_sphere(color=sp.Colors.Blue)
        augmented_points.apply_transform(sp.Transforms.Scale(0.01))
        augmented_points.enable_instancing(dataset._augment_points(item[0].numpy().T))
        augmented_points_frame.add_mesh(augmented_points)

        label = scene.create_label(
            text=text, color=sp.Colors.Black,
            size_in_pixels=80, offset_distance=0.6, camera_space=True)
        points_frame.add_label(label=label, position=[0.0, 0.0, -5.0])

    scene.save_as_html(str(output_file), title="ModelNet")
    webbrowser.open(str(output_file))


if __name__ == '__main__':
    typer.run(main)
