import os
import sys
import PyPeridyno as dyno
print(sys.path)

scn = dyno.SceneGraph()

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()

