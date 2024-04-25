import os

import PyPeridyno as dyno


def filePath(str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


scn = dyno.SceneGraph()
scn.set_lower_bound(dyno.Vector3f([-1.5, 0, -1.5]))
scn.set_upper_bound(dyno.Vector3f([1.5, 3, 1.5]))

mesh = dyno.StaticTriangularMesh3f()

FilePath = dyno.FilePath(filePath("cloth_shell/table/table.obj"))
mesh.var_file_name().set_value(FilePath)

boundary = dyno.VolumeBoundary3f()
boundary.load_cube(dyno.Vector3f([-1.5, 0, -1.5]), dyno.Vector3f([1.5, 3, 1.5]), 0.005, True)

boundary.load_sdf(filePath("cloth_shell/table/table.sdf"), False)

cloth = dyno.CodimensionalPD3f(0.3, 8000, 0.003, float('7e-4'), "default")
cloth.load_surface(filePath("cloth_shell/mesh40k_1_h90.obj"))
cloth.connect(boundary.import_triangular_system())

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.set_color(dyno.Color(0.4, 0.4, 1.0))

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.set_color(dyno.Color(0.8,0.8,0.8))
surfaceRenderer.var_use_vertex_normal().set_value(True)

cloth.state_triangle_set().connect(surfaceRendererCloth.in_triangle_set())
mesh.state_triangle_set().connect(surfaceRenderer.in_triangle_set())
cloth.graphics_pipeline().push_module(surfaceRendererCloth)
mesh.graphics_pipeline().push_module(surfaceRenderer)

cloth.set_visible(True)
mesh.set_visible(True)

scn.print_node_info(True)
scn.print_module_info(True)

scn.add_node(mesh)
scn.add_node(boundary)
scn.add_node(cloth)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
