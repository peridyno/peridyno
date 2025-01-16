import PyPeridyno as dyno
from PyPeridyno import Vector3f

scn = dyno.SceneGraph()
scn.set_lower_bound(Vector3f([-1.5, -1, -1.5]))
scn.set_upper_bound(Vector3f([1.5, 3, 1.5]))
scn.set_gravity(Vector3f([0,0,0]))

cloth = dyno.CodimensionalPD3f()
scn.add_node(cloth)
cloth.load_surface(dyno.get_asset_path() + "cloth_shell/cylinder400.obj")
cloth.set_dt(0.005)

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