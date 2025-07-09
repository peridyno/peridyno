import PyPeridyno as dyno
from PyPeridyno import Vector3f

scn = dyno.SceneGraph()
scn.setLowerBound(Vector3f([-5,0,-5]))
scn.setUpperBound(Vector3f([5,3,5]))

boundary = dyno.VolumeBoundary3f()
scn.addNode(boundary)
cloth = dyno.CodimensionalPD3f()
scn.addNode(cloth)

cloth.loadSurface(dyno.getAssetPath() + "cloth_shell/shootingCloth.obj")
cloth.connect(boundary.importTriangularSystems())

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