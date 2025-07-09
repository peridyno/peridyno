import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setLowerBound(dyno.Vector3f([-1.5, -1, -1.5]))
scn.setUpperBound(dyno.Vector3f([1.5, 3, 1.5]))
scn.setGravity(dyno.Vector3f([0, -200, 0]))

cloth = dyno.CodimensionalPD3f()
cloth.loadSurface(dyno.getAssetPath() + "cloth_shell/mesh_drop.obj")
scn.addNode(cloth)

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.setColor(dyno.Color(1, 1, 1))

cloth.stateTriangleSet().connect(surfaceRendererCloth.inTriangleSet())
cloth.graphicsPipeline().pushModule(surfaceRendererCloth)
cloth.setVisible(True)

scn.printNodeInfo(True)
scn.printSimulationInfo(True)



app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
