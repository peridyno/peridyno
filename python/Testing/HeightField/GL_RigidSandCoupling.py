import PyPeridyno as dyno

scn = dyno.SceneGraph()

jeep = dyno.Jeep3f()
scn.addNode(jeep)
jeep.varLocation().setValue(dyno.Vector3f([0, 0, -10]))

tank = dyno.Tank3f()
scn.addNode(tank)
tank.varLocation().setValue(dyno.Vector3f([-6, 0, -10]))

plane = dyno.PlaneModel3f()
scn.addNode(plane)
plane.varLengthX().setValue(40)
plane.varLengthZ().setValue(40)
plane.varSegmentX().setValue(20)
plane.varSegmentZ().setValue(20)

multibody = dyno.MultibodySystem3f()
scn.addNode(multibody)
plane.stateTriangleSet().connect(multibody.inTriangleSet())
jeep.connect(multibody.importVehicles())
tank.connect(multibody.importVehicles())

spacing = 0.1
res = 512
sand = dyno.GranularMedia3f()
scn.addNode(sand)
sand.varOrigin().setValue( dyno.Vector3f([-0.5 * res * spacing, 0, -0.5 * res * spacing]))
sand.varSpacing().setValue(spacing)
sand.varWidth().setValue(res)
sand.varHeight().setValue(res)
sand.varDepth().setValue(0.2)
sand.varDepthOfDiluteLayer().setValue(0.1)

coupling = dyno.RigidSandCoupling3f()
scn.addNode(coupling)
multibody.connect(coupling.importRigidBodySystem())
sand.connect(coupling.importGranularMedia())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
