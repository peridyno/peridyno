import PyPeridyno as dyno

scn = dyno.SceneGraph()

plane = dyno.PlaneModel3f()
scn.addNode(plane)
plane.varSegmentX().setValue(80)
plane.varSegmentZ().setValue(80)
plane.varLocation().setValue(dyno.Vector3f([0.0, 0.9, 0.0]))
plane.graphicsPipeline().disable()

sphereModel = dyno.SphereModel3f()
scn.addNode(sphereModel)
sphereModel.varLocation().setValue(dyno.Vector3f([0, 0.7, 0]))
sphereModel.varRadius().setValue(0.2)

sphere2vol = dyno.BasicShapeToVolume3f()
scn.addNode(sphere2vol)
sphere2vol.varGridSpacing().setValue(0.05)
sphereModel.connect(sphere2vol.importShape())



cloth = dyno.Cloth3f()
scn.addNode(cloth)
cloth.setDt(0.001)
plane.stateTriangleSet().connect(cloth.inTriangleSet())

boundary = dyno.VolumeBoundary3f()
scn.addNode(boundary)
sphere2vol.connect(boundary.importVolumes())
cloth.connect(boundary.importTriangularSystems())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(800, 600, True)
app.mainLoop()
