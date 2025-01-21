import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_gravity(dyno.Vector3f([0,-9.8,0]))

gear = dyno.Gear3f()
scn.add_node(gear)

plane = dyno.PlaneModel3f()
scn.add_node(plane)
plane.var_length_x().set_value(50)
plane.var_length_z().set_value(50)
plane.var_segment_x().set_value(10)
plane.var_segment_z().set_value(10)

convoy = dyno.MultibodySystem3f()
scn.add_node(convoy)
gear.connect(convoy.import_vehicles())

plane.state_triangle_set().connect(convoy.in_triangle_set())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
