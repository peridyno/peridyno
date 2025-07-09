import PyPeridyno as dyno

scn = dyno.SceneGraph()

oceanPatch = dyno.OceanPatch3f()
scn.addNode(oceanPatch)
oceanPatch.varWindType().setValue(8)

root = dyno.Ocean3f()
scn.addNode(root)
root.varExtentX().setValue(2)
root.varExtentZ().setValue(2)
oceanPatch.connect(root.importOceanPatch())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
