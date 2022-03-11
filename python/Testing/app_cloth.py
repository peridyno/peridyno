import PyPeridyno as dyno

scn = dyno.SceneGraph()

cloth = dyno.Cloth3f()
cloth.load_particles("../../data/cloth/cloth.obj")
cloth.load_surface("../../data/cloth/cloth.obj")
scn.add_node(cloth)

boundary = dyno.StaticBoundary3f()
boundary.load_cube(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([1.0, 1.0, 1.0]), 0.005, True)
boundary.load_sphere(dyno.Vector3f([0.5, 0.7, 0.5]), 0.08, 0.005, False, True)

boundary.add_particle_system(cloth)
scn.add_node(boundary)


pointRender = dyno.GLPointVisualModule3f()
pointRender.set_color(dyno.Vector3f([1, 0.2, 1]))
pointRender.set_colorMapMode(pointRender.ColorMapMode.PER_OBJECT_SHADER)
cloth.current_topology().connect(pointRender.in_pointSet())
cloth.state_velocity().connect(pointRender.in_color())

cloth.graphics_pipeline().push_module(pointRender)
cloth.set_visible(True)

surfaceRenderer = dyno.GLSurfaceVisualModule3f()
cloth.current_topology().connect(surfaceRenderer.in_triangleSet())
cloth.graphics_pipeline().push_module(surfaceRenderer)


app = dyno.GLApp()
app.set_scenegraph(scn)
app.create_window(1024, 768)
app.main_loop()