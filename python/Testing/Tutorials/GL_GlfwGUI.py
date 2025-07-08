import os
import sys
import PyPeridyno as dyno
print(sys.path)

scn = dyno.SceneGraph()

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()

