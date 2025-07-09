import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setGravity(dyno.Vector3f([0,-9.8,0]))

gear = dyno.Gear3f()
scn.addNode(gear)

plane = dyno.PlaneModel3f()
scn.addNode(plane)
plane.varLengthX().setValue(50)
plane.varLengthZ().setValue(50)
plane.varSegmentX().setValue(10)
plane.varSegmentZ().setValue(10)

convoy = dyno.MultibodySystem3f()
scn.addNode(convoy)
gear.connect(convoy.importVehicles())

plane.stateTriangleSet().connect(convoy.inTriangleSet())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()
