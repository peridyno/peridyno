import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setGravity(dyno.Vector3f([0, -9.8, 0]))

emitter = dyno.CircularEmitter3f()
scn.addNode(emitter)
emitter.varLocation().setValue(dyno.Vector3f([0, 1, 0]))

plane = dyno.PlaneModel3f()
scn.addNode(plane)
plane.varLocation().setValue(dyno.Vector3f([2,0,2]))

sphere = dyno.SphereModel3f()
scn.addNode(sphere)
sphere.varLocation().setValue(dyno.Vector3f([0,0.5,0]))
sphere.varScale().setValue(dyno.Vector3f([0.2,0.2,0.2]))

merge = dyno.MergeTriangleSet3f()
scn.addNode(merge)
plane.stateTriangleSet().connect(merge.inFirst())
sphere.stateTriangleSet().connect(merge.inSecond())

sfi = dyno.SemiAnalyticalSFINode3f()
scn.addNode(sfi)

emitter.connect(sfi.importParticleEmitters())
merge.stateTriangleSet().connect(sfi.inTriangleSet())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
# app.renderWindow().getCamera().setUnitScale(512)
app.mainLoop()
