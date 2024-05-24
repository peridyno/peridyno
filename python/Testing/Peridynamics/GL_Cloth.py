import PyPeridyno as dyno

scn = dyno.SceneGraph()

plane = dyno.PlaneModel3f()
plane.var_segment_x().set_value(80)
plane.var_segment_z().set_value(80)
plane.var_location().set_value(dyno.Vector3f([0.0, 0.9, 0.0]))
plane.graphics_pipeline().disable()

cloth = dyno.Cloth3f()
cloth.set_dt(0.001)
plane.state_triangle_set().connect(cloth.in_triangle_set())

root = dyno.VolumeBoundary3f()
root.load_shpere(dyno.Vector3f([0.0, 0.7, 0.0]), 0.08, 0.005, False, True)

cloth.connect(root.import_triangular_systems())

pointRenderer = dyno.GLPointVisualModule()
pointRenderer.set_color(dyno.Color(1, 0.2, 1))
pointRenderer.set_color_map_mode(pointRenderer.ColorMapMode.PER_VERTEX_SHADER)
pointRenderer.var_point_size().set_value(0.002)
cloth.state_triangle_set().connect(pointRenderer.in_point_set())
cloth.state_velocity().connect(pointRenderer.in_color())

cloth.graphics_pipeline().push_module(pointRenderer)
cloth.set_visible(True)

wireRenderer = dyno.GLWireframeVisualModule()
wireRenderer.var_base_color().set_value(dyno.Color(1.0, 0.8, 0.8))
wireRenderer.var_radius().set_value(0.001)
wireRenderer.var_render_mode().set_current_key(wireRenderer.EEdgeMode.CYLINDER)
cloth.state_triangle_set().connect(wireRenderer.in_edge_set())
cloth.graphics_pipeline().push_module(wireRenderer)

surfaceRenderer = dyno.GLSurfaceVisualModule()
cloth.state_triangle_set().connect(surfaceRenderer.in_triangle_set())
cloth.graphics_pipeline().push_module(surfaceRenderer)

scn.add_node(plane)
scn.add_node(cloth)
scn.add_node(root)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(800, 600, True)
app.main_loop()
