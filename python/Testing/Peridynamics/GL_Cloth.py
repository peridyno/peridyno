import PyPeridyno as dyno

scn = dyno.SceneGraph()

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(800, 600, True)
app.main_loop()