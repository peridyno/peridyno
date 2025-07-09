import PyPeridyno as dyno

scn = dyno.SceneGraph()

patch = dyno.OceanPatch3f()
scn.addNode(patch)
patch.varWindType().setValue(8)

ocean = dyno.LargeOcean3f()
scn.addNode(ocean)

patch.connect(ocean.importOceanPatch())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
