import os
import sys
os.add_dll_directory("E:\\IDE\\qt6\\6.5.3\\msvc2019_64\\bin")
import PyPeridyno as dyno
print(sys.path)

scn = dyno.SceneGraph()

app = dyno.QtApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()

