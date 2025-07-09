import PyPeridyno as dyno

scn = dyno.SceneGraph()

sand = dyno.GranularMedia3f()
scn.addNode(sand)
sand.varOrigin().setValue(dyno.Vector3f([-32, 0, -32]))


tracking = dyno.SurfaceParticleTracking3f()
scn.addNode(tracking)
sand.connect(tracking.importGranularMedia())


app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
