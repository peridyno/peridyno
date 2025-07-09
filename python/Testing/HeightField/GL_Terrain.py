import PyPeridyno as dyno

scn = dyno.SceneGraph()

land = dyno.LandScape3f()
land.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "landscape/Landscape_1_Map_1024x1024.png"))
land.varLocation().setValue(dyno.Vector3f([0, 100, 0]))
land.varScale().setValue(dyno.Vector3f([1, 64, 1]))

scn.addNode(land)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)

app.mainLoop()
