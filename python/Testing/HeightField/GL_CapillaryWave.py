import PyPeridyno as dyno

scn = dyno.SceneGraph()

root = dyno.CapillaryWave3f()
scn.add_node(root)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
#app.render_window().get_camera().set_unit_scale(20)
app.main_loop()
