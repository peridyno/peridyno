import PyPeridyno as dyno

totalScale = 6

scn = dyno.SceneGraph()

jeep = dyno.Jeep3f()
scn.addNode(jeep)

multibody = dyno.MultibodySystem3f()
scn.addNode(multibody)
jeep.connect(multibody.importVehicles())
jeep.varLocation().setValue(dyno.Vector3f([0, 1, -5]))

ObjLand = dyno.ObjLoader3f()
scn.addNode(ObjLand)
ObjLand.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "landscape/Landscape_resolution_1000_1000.obj"))
ObjLand.varScale().setValue(dyno.Vector3f([6, 6, 6]))
ObjLand.varLocation().setValue(dyno.Vector3f([0, 0, 0.5]))
glLand = ObjLand.graphicsPipeline().findFirstModule()
glLand.varBaseColor().setValue(dyno.Color(0.82745, 0.82745, 0.82745))
glLand.varUseVertexNormal().setValue(True)

ObjLand.outTriangleSet().connect(multibody.inTriangleSet())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1366, 768, True)
app.mainLoop()
