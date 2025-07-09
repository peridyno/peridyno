import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setLowerBound(dyno.Vector3f([-1.5, 0, -1.5]))
scn.setUpperBound(dyno.Vector3f([1.5, 3, 1.5]))

object = dyno.StaticMeshLoader3f()
scn.addNode(object)
FilePath = dyno.FilePath(dyno.getAssetPath() + "cloth_shell/table/table.obj")
object.varFileName().setValue(FilePath, True)

volLoader = dyno.VolumeLoader3f()
scn.addNode(volLoader)
FilePath = dyno.FilePath(dyno.getAssetPath() + "cloth_shell/table/table.sdf")
volLoader.varFileName().setValue(FilePath, True)

boundary = dyno.VolumeBoundary3f()
scn.addNode(boundary)
volLoader.connect(boundary.importVolumes())

cloth = dyno.CodimensionalPD3f()
scn.addNode(cloth)
cloth.loadSurface(dyno.getAssetPath() + "cloth_shell/mesh40k_1_h90.obj")
cloth.connect(boundary.importTriangularSystems())

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.setColor(dyno.Color(0.4, 0.4, 1.0))

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.setColor(dyno.Color(0.8, 0.8, 0.8))
surfaceRenderer.varUseVertexNormal().setValue(True, True)

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
