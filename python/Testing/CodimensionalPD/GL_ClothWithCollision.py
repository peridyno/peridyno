import PyPeridyno as dyno
from PyPeridyno import Vector3f

scn = dyno.SceneGraph()
scn.setLowerBound(Vector3f([-1.5, 0, -1.5]))
scn.setUpperBound(Vector3f([1.5, 3, 1.5]))

object = dyno.StaticMeshLoader3f()
scn.addNode(object)
object.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "cloth_shell/model_ball.obj"))

volLoader = dyno.VolumeLoader3f()
scn.addNode(volLoader)
volLoader.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "cloth_shell/model_sdf.sdf"))

boundary = dyno.VolumeBoundary3f()
scn.addNode(boundary)
volLoader.connect(boundary.importVolumes())

cloth = dyno.CodimensionalPD3f()
scn.addNode(cloth)
cloth.loadSurface(dyno.getAssetPath() + "cloth_shell/mesh_120.obj")
cloth.connect(boundary.importTriangularSystems())
cloth.setDt(0.001)

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.setColor(dyno.Color(0.4,0.4,0.4))

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.setColor(dyno.Color(0.4,0.4,0.4))
surfaceRenderer.varUseVertexNormal().setValue(True)

cloth.stateTriangleSet().connect(surfaceRendererCloth.inTriangleSet())
object.stateTriangleSet().connect(surfaceRenderer.inTriangleSet())
cloth.graphicsPipeline().pushModule(surfaceRendererCloth)
object.graphicsPipeline().pushModule(surfaceRenderer)
cloth.setVisible(True)
object.setVisible(True)

scn.printNodeInfo(True)
scn.printSimulationInfo(True)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()