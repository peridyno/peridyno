import PyPeridyno as dyno

scn = dyno.SceneGraph()

jeep = dyno.Jeep3f()
scn.add_node(jeep)
jeep.var_location().set_value(dyno.Vector3f([0, 0, -10]))

tank = dyno.Tank3f()
scn.add_node(tank)
tank.var_location().set_value(dyno.Vector3f([-6, 0, -10]))

plane = dyno.PlaneModel3f()
scn.add_node(plane)
plane.var_length_x().set_value(40)
plane.var_length_z().set_value(40)
plane.var_segment_x().set_value(20)
plane.var_segment_z().set_value(20)

multibody = dyno.MultibodySystem3f()
scn.add_node(multibody)
plane.state_triangle_set().connect(multibody.in_triangle_set())
jeep.connect(multibody.import_vehicles())
tank.connect(multibody.import_vehicles())

spacing = 0.1
res = 512
sand = dyno.GranularMedia3f()
scn.add_node(sand)
sand.var_origin().set_value( dyno.Vector3f([-0.5 * res * spacing, 0, -0.5 * res * spacing]))
sand.var_spacing().set_value(spacing)
sand.var_width().set_value(res)
sand.var_height().set_value(res)
sand.var_depth().set_value(0.2)
sand.var_depth_of_dilute_layer().set_value(0.1)

coupling = dyno.RigidSandCoupling3f()
scn.add_node(coupling)
multibody.connect(coupling.import_rigid_body_system())
sand.connect(coupling.import_granular_media())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
