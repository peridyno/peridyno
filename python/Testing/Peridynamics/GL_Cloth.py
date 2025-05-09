import PyPeridyno as dyno

scn = dyno.SceneGraph()

plane = dyno.PlaneModel3f()
scn.add_node(plane)
plane.var_segment_x().set_value(80)
plane.var_segment_z().set_value(80)
plane.var_location().set_value(dyno.Vector3f([0.0, 0.9, 0.0]))
plane.graphics_pipeline().disable()

sphereModel = dyno.SphereModel3f()
scn.add_node(sphereModel)
sphereModel.var_location().set_value(dyno.Vector3f([0, 0.7, 0]))
sphereModel.var_radius().set_value(0.2)

sphere2vol = dyno.BasicShapeToVolume3f()
scn.add_node(sphere2vol)
sphere2vol.var_grid_spacing().set_value(0.05)
sphereModel.connect(sphere2vol.import_shape())



cloth = dyno.Cloth3f()
scn.add_node(cloth)
cloth.set_dt(0.001)
plane.state_triangle_set().connect(cloth.in_triangle_set())

boundary = dyno.VolumeBoundary3f()
scn.add_node(boundary)
sphere2vol.connect(boundary.import_volumes())
cloth.connect(boundary.import_triangular_systems())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(800, 600, True)
app.main_loop()
