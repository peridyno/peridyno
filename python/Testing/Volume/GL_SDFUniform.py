import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([2,2,2]))
scn.set_lower_bound(dyno.Vector3f([-2,-2,-2]))

sphere = dyno.SphereModel3f()
scn.add_node(sphere)

volume = dyno.VolumeGenerator3f()
scn.add_node(volume)
volume.var_padding().set_value(10)
volume.var_spacing().set_value(0.05)

sphere.state_triangle_set().connect(volume.in_triangle_set())

clipper = dyno.VolumeClipper3f()
scn.add_node(clipper)
volume.state_level_set().connect(clipper.in_level_set())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
