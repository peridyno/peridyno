import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

jeep = dyno.Jeep3f()
scn.addNode(jeep)
jeep.varLocation().setValue(dyno.Vector3f([4,0,0]))

tank = dyno.Tank3f()
scn.addNode(tank)

plane = dyno.PlaneModel3f()
scn.addNode(plane)
plane.varLengthX().setValue(50)
plane.varLengthZ().setValue(50)
plane.varSegmentX().setValue(10)
plane.varSegmentZ().setValue(10)

convoy = dyno.MultibodySystem3f()
scn.addNode(convoy)
jeep.connect(convoy.importVehicles())
tank.connect(convoy.importVehicles())

plane.stateTriangleSet().connect(convoy.inTriangleSet())

mapper = dyno.DiscreteElementsToTriangleSet3f()
convoy.stateTopology().connect(mapper.inDiscreteElements())
convoy.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(1,1,0))
sRender.setAlpha(0.5)
mapper.outTriangleSet().connect(sRender.inTriangleSet())
convoy.graphicsPipeline().pushModule(sRender)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()
