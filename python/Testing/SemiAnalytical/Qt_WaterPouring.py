import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_gravity(dyno.Vector3f([0, -9.8, 0]))

emitter = dyno.CircularEmitter3f()
scn.add_node(emitter)
emitter.var_location().set_value(dyno.Vector3f([0, 1, 0]))

plane = dyno.PlaneModel3f()
scn.add_node(plane)
plane.var_location().set_value(dyno.Vector3f([2,0,2]))

sphere = dyno.SphereModel3f()
scn.add_node(sphere)
sphere.var_location().set_value(dyno.Vector3f([0,0.5,0]))
sphere.var_scale().set_value(dyno.Vector3f([0.2,0.2,0.2]))

merge = dyno.MergeTriangleSet3f()
scn.add_node(merge)
plane.state_triangle_set().connect(merge.in_first())
sphere.state_triangle_set().connect(merge.in_second())

sfi = dyno.SemiAnalyticalSFINode3f()
scn.add_node(sfi)

emitter.connect(sfi.import_particle_emitters())
merge.state_triangle_set().connect(sfi.in_triangle_set())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
