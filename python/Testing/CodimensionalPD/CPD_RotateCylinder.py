import PyPeridyno as dyno
from PyPeridyno import Vector3f

scn = dyno.SceneGraph()
scn.setLowerBound(Vector3f([-1.5, -1, -1.5]))
scn.setUpperBound(Vector3f([1.5, 3, 1.5]))
scn.setGravity(Vector3f([0,0,0]))

cloth = dyno.CodimensionalPD3f()
scn.addNode(cloth)
cloth.loadSurface(dyno.getAssetPath() + "cloth_shell/cylinder400.obj")
cloth.setDt(0.005)

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.setColor(dyno.Color(1,1,1))

cloth.stateTriangleSet().connect(surfaceRendererCloth.inTriangleSet())
cloth.graphicsPipeline().pushModule(surfaceRendererCloth)
cloth.setVisible(True)

scn.printNodeInfo(True)
scn.printSimulationInfo(True)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()