import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

tank = dyno.Tank3f()
scn.add_node(tank)

plane = dyno.PlaneModel3f()
scn.add_node(plane)
plane.var_length_x().set_value(50)
plane.var_length_z().set_value(50)
plane.var_segment_x().set_value(10)
plane.var_segment_z().set_value(10)

convoy = dyno.MultibodySystem3f()
scn.add_node(convoy)
tank.connect(convoy.import_vehicles())

plane.state_triangle_set().connect(convoy.in_triangle_set())

mapper = dyno.DiscreteElementsToTriangleSet3f()
convoy.state_topology().connect(mapper.in_discrete_elements())
convoy.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(1,1,0))
sRender.set_alpha(0.5)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
convoy.graphics_pipeline().push_module(sRender)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
