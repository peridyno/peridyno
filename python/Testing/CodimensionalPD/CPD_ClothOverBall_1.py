import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_lower_bound(dyno.Vector3f([-1.5, 0, -1.5]))
scn.set_upper_bound(dyno.Vector3f([1.5, 3, 1.5]))

object = dyno.StaticMeshLoader3f()
scn.add_node(object)
object.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "cloth_shell/ball/ball_model.obj"))

volLoader = dyno.VolumeLoader3f()
scn.add_node(volLoader)
volLoader.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "cloth_shell/ball/ball_small_size_15.sdf"))

boundary = dyno.VolumeBoundary3f()
scn.add_node(boundary)
volLoader.connect(boundary.import_volumes())

cloth = dyno.CodimensionalPD3f()
scn.add_node(cloth)
cloth.set_dt(0.001)
cloth.load_surface(dyno.FilePath(dyno.get_asset_path() + "cloth_shell/cloth_size_17_alt/cloth_40k_1.obj"))
cloth.connect(boundary.import_triangular_systems())

solver = cloth.animation_pipeline().find_first_module

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.set_color(dyno.Color(0.4, 0.4, 1.0))

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.set_color(dyno.Color(0.4, 0.4, 0.4))
surfaceRenderer.var_use_vertex_normal().set_value(True)
cloth.state_triangle_set().connect(surfaceRendererCloth.in_triangle_set())
object.state_triangle_set().connect(surfaceRenderer.in_triangle_set())
cloth.graphics_pipeline().push_module(surfaceRendererCloth)
object.graphics_pipeline().push_module(surfaceRenderer)
cloth.set_visible(True)
object.set_visible(True)
scn.print_node_info(True)
scn.print_simulation_info(True)


app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
