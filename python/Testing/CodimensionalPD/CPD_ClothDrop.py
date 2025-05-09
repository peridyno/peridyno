import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_lower_bound(dyno.Vector3f([-1.5, -1, -1.5]))
scn.set_upper_bound(dyno.Vector3f([1.5, 3, 1.5]))
scn.set_gravity(dyno.Vector3f([0, -200, 0]))

cloth = dyno.CodimensionalPD3f()

cloth.load_surface(dyno.get_asset_path() + "cloth_shell/mesh_drop.obj")

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.set_color(dyno.Color(1, 1, 1))

cloth.state_triangle_set().connect(surfaceRendererCloth.in_triangle_set())
cloth.graphics_pipeline().push_module(surfaceRendererCloth)
cloth.set_visible(True)

scn.print_node_info(True)
scn.print_simulation_info(True)

scn.add_node(cloth)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
