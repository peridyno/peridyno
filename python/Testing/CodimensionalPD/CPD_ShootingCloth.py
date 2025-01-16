import PyPeridyno as dyno
from PyPeridyno import Vector3f

scn = dyno.SceneGraph()
scn.set_lower_bound(Vector3f([-5,0,-5]))
scn.set_upper_bound(Vector3f([5,3,5]))

boundary = dyno.VolumeBoundary3f()
scn.add_node(boundary)
cloth = dyno.CodimensionalPD3f()
scn.add_node(cloth)

cloth.load_surface(dyno.get_asset_path() + "cloth_shell/shootingCloth.obj")
cloth.connect(boundary.import_triangular_systems())

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.set_color(dyno.Color(1,1,1))

cloth.state_triangle_set().connect(surfaceRendererCloth.in_triangle_set())
cloth.graphics_pipeline().push_module(surfaceRendererCloth)
cloth.set_visible(True)

scn.print_node_info(True)
scn.print_simulation_info(True)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()