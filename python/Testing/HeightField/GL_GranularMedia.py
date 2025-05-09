import PyPeridyno as dyno

scn = dyno.SceneGraph()

sand = dyno.GranularMedia3f()
scn.add_node(sand)
sand.var_origin().set_value(dyno.Vector3f([-32, 0, -32]))


tracking = dyno.SurfaceParticleTracking3f()
scn.add_node(tracking)
sand.connect(tracking.import_granular_media())


app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
