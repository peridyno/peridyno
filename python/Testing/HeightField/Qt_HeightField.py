import QtPathHelper
import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.addNode(dyno.OceanPatch3f())


app = dyno.QtApp()
#app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.renderWindow().getCamera().setUnitScale(52)
app.mainLoop()
