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
scene.set_lower_bound(dyno.Vector3f([-1.5, 0, -1.5]))
scene.set_upper_bound(dyno.Vector3f([1.5, 3, 1.5]))
scene.set_gravity(dyno.Vector3f([0, -0.98, 0]))

staticTriangularMesh = dyno.StaticTriangularMesh3f()
staticTriangularMesh.var_file_name().set_value(dyno.FilePath(filePath("cloth_shell/model_ball.obj")))

boundary = dyno.VolumeBoundary3f()
boundary.load_cube(dyno.Vector3f([-1.5, 0, -1.5]), dyno.Vector3f([1.5, 3, 1.5]), 0.005, True)
boundary.load_sdf(filePath("cloth_shell/model_sdf.sdf"), False)

cloth = dyno.CodimensionalPD3f(0.15, 20, 0, 0.001, "default")
cloth.load_surface(filePath("cloth_shell/mesh_120.obj"))
cloth.connect(boundary.import_triangular_systems())
cloth.set_dt(0.001)
cloth.set_grad_ite_eps(0.0001)
cloth.set_max_ite_number(10)
cloth.set_accelerated(True)

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.set_color(dyno.Color(0.4, 0.4, 1.0))
# surfaceRendererCloth.var_use_vertex_normal().set_value(True)

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.set_color(dyno.Color(0.4, 0.4, 0.4))
surfaceRenderer.var_use_vertex_normal().set_value(True)

cloth.state_triangle_set().connect(surfaceRendererCloth.in_triangle_set())
staticTriangularMesh.state_triangle_set().connect(surfaceRenderer.in_triangle_set())
cloth.graphics_pipeline().push_module(surfaceRendererCloth)
staticTriangularMesh.graphics_pipeline().push_module(surfaceRenderer)
cloth.set_visible(True)
staticTriangularMesh.set_visible(True)

scene.print_node_info(True)
scene.print_module_info(True)

scene.add_node(staticTriangularMesh)
scene.add_node(boundary)
scene.add_node(cloth)