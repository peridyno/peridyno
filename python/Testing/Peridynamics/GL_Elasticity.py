import PyPeridyno as dyno

scn = dyno.SceneGraph()

root = dyno.StaticBoundary3f()
root.load_cube(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([1, 1, 1]), 0.005, True)

bunny = dyno.ElasticBody3f()
bunny.connect(root.import_particle_systems())

bunny.load_particles(dyno.get_asset_path() + "bunny/bunny_points.obj")
bunny.scale(1.0)
bunny.translate(dyno.Vector3f([0.5, 0.1, 0.5]))
bunny.set_visible(True)

pointRenderer = dyno.GLPointVisualModule()
pointRenderer.set_color(dyno.Color(1, 0.2, 1))
pointRenderer.set_color_map_mode(pointRenderer.ColorMapMode.PER_OBJECT_SHADER)
bunny.state_point_set().connect(pointRenderer.in_point_set())
bunny.state_velocity().connect(pointRenderer.in_color())
bunny.graphics_pipeline().push_module(pointRenderer)

scn.add_node(root)
scn.add_node(bunny)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
