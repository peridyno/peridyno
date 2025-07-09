import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([2,2,2]))
scn.setLowerBound(dyno.Vector3f([-2,-2,-2]))

sphere = dyno.SphereModel3f()
scn.addNode(sphere)

volume = dyno.VolumeGenerator3f()
scn.addNode(volume)
volume.varPadding().setValue(10)
volume.varSpacing().setValue(0.05)

sphere.stateTriangleSet().connect(volume.inTriangleSet())

clipper = dyno.VolumeClipper3f()
scn.addNode(clipper)
volume.stateLevelSet().connect(clipper.inLevelSet())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
# app.renderWindow().getCamera().setUnitScale(512)
app.mainLoop()
