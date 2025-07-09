import PyPeridyno as dyno

scn = dyno.SceneGraph()

root = dyno.CapillaryWave3f()
scn.addNode(root)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
#app.renderWindow().getCamera().setUnitScale(20)
app.mainLoop()
