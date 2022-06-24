import random
import pathlib
import webbrowser

import typer
import trimesh
import scenepic as sp

from dataset import ModelNet


def main(
        dataset_dir: pathlib.Path = pathlib.Path("/home/szobov/dev/learning/pointnet/dataset/ModelNet40"),
        output_file: pathlib.Path = pathlib.Path("/tmp/modelnet.html")
):
    dataset = ModelNet(dataset_dir)
    scene = sp.Scene()
    mesh_canvas = scene.create_canvas_3d(
        "mesh", width=400, height=400,
        shading=sp.Shading(bg_color=sp.Colors.White))
    points_canvas = scene.create_canvas_3d(
        "points ", width=400, height=400,
        shading=sp.Shading(bg_color=sp.Colors.White))
    scene.link_canvas_events(mesh_canvas, points_canvas)
    for indx in random.choices(range(len(dataset)), k=10):
        item = dataset[indx]
        text = dataset.get_class_description(int(item[1][0]))
        mesh_frame = mesh_canvas.create_frame()
        points_frame = points_canvas.create_frame()
        mesh = scene.create_mesh()
        for face in trimesh.load(dataset.files[indx]).triangles:
            mesh.add_triangle(color=sp.Colors.Red,
                              p0=face[0], p1=face[1], p2=face[2])
        mesh.apply_transform(sp.Transforms.Scale(0.01))
        mesh_frame.add_mesh(mesh)
        points = scene.create_mesh()
        points.add_sphere(color=sp.Colors.Green)
        points.apply_transform(sp.Transforms.Scale(0.01))
        points.enable_instancing(item[0] / 100)
        points_frame.add_mesh(points)
        label = scene.create_label(
            text=text, color=sp.Colors.Black,
            size_in_pixels=80, offset_distance=0.6, camera_space=True)
        points_frame.add_label(label=label, position=[0.0, 0.0, -5.0])

    scene.save_as_html(str(output_file), title="ModelNet")
    webbrowser.open(str(output_file))


if __name__ == '__main__':
    typer.run(main)
