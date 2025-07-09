import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setLowerBound(dyno.Vector3f([-1.5, 0, -1.5]))
scn.setUpperBound(dyno.Vector3f([1.5, 3, 1.5]))
scn.setGravity(dyno.Vector3f([0, -0.98, 0]))

object = dyno.StaticMeshLoader3f()
scn.addNode(object)
object.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "cloth_shell/v1/woman_model.obj"))

volLoader = dyno.VolumeLoader3f()
scn.addNode(volLoader)
volLoader.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "cloth_shell/v1/woman_v1.sdf"))

boundary = dyno.VolumeBoundary3f()
scn.addNode(boundary)

cloth = dyno.CodimensionalPD3f()
scn.addNode(cloth)
cloth.loadSurface(dyno.getAssetPath() + "cloth_shell/v1/cloth_highMesh.obj")
cloth.connect(boundary.importTriangularSystems())
cloth.setDt(5e-4)

# interaction = dyno.DragVertexInteraction3f()
# interaction.varCacheEvent().setValue(False)
# cloth.stateTriangleSet().connect(interaction.inInitialTriangleSet())
# cloth.statePosition().connect(interaction.inPosition())
# cloth.stateVelocity().connect(interaction.inVelocity())
# cloth.stateAttribute().connect(interaction.inAttribute())
# cloth.stateTimeStep().connect(interaction.inTimeStep())
# cloth.animationPipeline().pushModule(interaction)

surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.setColor(dyno.Color(0.4, 0.4, 1.0))

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.setColor(dyno.Color(0.4, 0.4, 0.4))
surfaceRenderer.varUseVertexNormal().setValue(True)
cloth.stateTriangleSet().connect(surfaceRendererCloth.inTriangleSet())
object.stateTriangleSet().connect(surfaceRenderer.inTriangleSet())
cloth.graphicsPipeline().pushModule(surfaceRendererCloth)
object.graphicsPipeline().pushModule(surfaceRenderer)
cloth.setVisible(True)
object.setVisible(True)
scn.printNodeInfo(True)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
