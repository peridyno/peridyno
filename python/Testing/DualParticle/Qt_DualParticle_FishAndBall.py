import PyPeridyno as dyno
from PyPeridyno import Vector3f

scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([3, 3, 3]))
scn.setLowerBound(dyno.Vector3f([-3, -3, -3]))

ptsLoader = dyno.PointsLoader3f()
scn.addNode(ptsLoader)
ptsLoader.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "fish/FishPoints.obj"))
ptsLoader.varRotation().setValue(dyno.Vector3f([0, 3.14 * 2 / 5, 0]))
ptsLoader.varLocation().setValue(dyno.Vector3f([0, 0.4, 0.3]))
initialParticles = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles)
ptsLoader.outPointSet().promoteOutput().connect(initialParticles.inPoints())

fluid = dyno.DualParticleFluid3f()
scn.addNode(fluid)
fluid.varReshuffleParticles().setValue(True)
initialParticles.connect(fluid.importInitialStates())

ball = dyno.SphereModel3f()
scn.addNode(ball)
ball.varScale().setValue(Vector3f([0.38, 0.38, 0.38]))
ball.varLocation().setValue(Vector3f([0, 0, 0.3]))
sRenderf = dyno.GLSurfaceVisualModule()
sRenderf.setColor(dyno.Color(0.8, 0.52, 0.25))
sRenderf.setVisible(True)
sRenderf.varUseVertexNormal().setValue(True)
ball.stateTriangleSet().connect(sRenderf.inTriangleSet())
ball.graphicsPipeline().pushModule(sRenderf)

pmCollide = dyno.TriangularMeshBoundary3f()
scn.addNode(pmCollide)
ball.stateTriangleSet().connect(pmCollide.inTriangleSet())
fluid.connect(pmCollide.importParticleSystems())

calculateNorm = dyno.CalculateNorm3f()
fluid.stateVelocity().connect(calculateNorm.inVec())
fluid.graphicsPipeline().pushModule(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.varMax().setValue(5.0)
calculateNorm.outNorm().connect(colorMapper.inScalar())
fluid.graphicsPipeline().pushModule(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.setColor(dyno.Color(1, 0, 0))

ptRender.setColorMapMode(ptRender.ColorMapMode.PER_VERTEX_SHADER)
fluid.statePointSet().connect(ptRender.inPointSet())
colorMapper.outColor().connect(ptRender.inColor())
fluid.graphicsPipeline().pushModule(ptRender)

# A simple color bar widget for node
colorBar = dyno.ImColorbar3f()
colorBar.varMax().setValue(5.0)
colorBar.varFieldName().setValue("Velocity")
calculateNorm.outNorm().connect(colorBar.inScalar())
# add the widget to app
fluid.graphicsPipeline().pushModule(colorBar)

vpRender = dyno.GLPointVisualModule()
vpRender.setColor(dyno.Color(1, 1, 0))
vpRender.setColorMapMode(vpRender.ColorMapMode.PER_VERTEX_SHADER)
fluid.state_virtual_pointSet().connect(vpRender.inPointSet())
vpRender.varPointSize().setValue(0.0005)
fluid.graphicsPipeline().pushModule(vpRender)


app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
