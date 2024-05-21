import os

import PyPeridyno as dyno


def filePath(str):
    script_dir = os.getcwd()
    relative_path = "../../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


scene = dyno.SceneGraph()
scene.set_lower_bound(dyno.Vector3f([-1.5, -1, -1.5]))
scene.set_upper_bound(dyno.Vector3f([1.5, 3, 1.5]))
scene.set_gravity(dyno.Vector3f([0, -200, 0]))

cloth = dyno.CodimensionalPD3f(0.15, 120, 0.001, 0.0001, "default")

cloth.load_surface(filePath("cloth_shell/mesh_drop.obj"))


surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.set_color(dyno.Color(1, 1, 1))

cloth.state_triangle_set().connect(surfaceRendererCloth.in_triangle_set())
cloth.graphics_pipeline().push_module(surfaceRendererCloth)
cloth.set_visible(True)

scene.print_node_info(True)
scene.print_module_info(True)
scene.add_node(cloth)