import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setTotalTime(3)
scn.setGravity(dyno.Vector3f([0, -9.8, 0]))
scn.setLowerBound(dyno.Vector3f([-1, 0, 0]))
scn.setUpperBound(dyno.Vector3f([1, 1, 1]))

emitter = dyno.SquareEmitter3f()
scn.addNode(emitter)
emitter.varLocation().setValue(dyno.Vector3f([0, 0.5, 0.5]))

fluid = dyno.ParticleFluid3f()
scn.addNode(fluid)
emitter.connect(fluid.importParticleEmitters())

ptRender = dyno.GLPointVisualModule()
ptRender.varPointSize().setValue(0.002)
ptRender.setColor(dyno.Color(1, 0, 0))
ptRender.setColorMapMode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

calculateNorm = dyno.CalculateNorm3f()
colorMapper = dyno.ColorMapping3f()
colorMapper.varMax().setValue(5)
fluid.stateVelocity().connect(calculateNorm.inVec())
calculateNorm.outNorm().connect(colorMapper.inScalar())

colorMapper.outColor().connect(ptRender.inColor())
fluid.statePointSet().connect(ptRender.inPointSet())

fluid.graphicsPipeline().pushModule(calculateNorm)
fluid.graphicsPipeline().pushModule(colorMapper)
fluid.graphicsPipeline().pushModule(ptRender)

barricade = dyno.StaticMeshLoader3f()
scn.addNode(barricade)
barricade.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "bowl/barricade.obj"))
barricade.varLocation().setValue(dyno.Vector3f([0.1, 0.022, 0.5]))

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(0.8, 0.52, 0.25))
sRender.setVisible(True)
sRender.varUseVertexNormal().setValue(True)
barricade.stateTriangleSet().connect(sRender.inTriangleSet())
barricade.graphicsPipeline().pushModule(sRender)

boundary = dyno.StaticMeshLoader3f()
scn.addNode(boundary)
boundary.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "standard/standardCube2.obj"))
boundary.graphicsPipeline().disable()

sfi = dyno.TriangularMeshBoundary3f()
scn.addNode(sfi)
pbd = dyno.SemiAnalyticalPositionBasedFluidModel3f()
pbd.varSmoothingLength().setValue(0.0085)

merge = dyno.MergeTriangleSet3f()
scn.addNode(merge)
boundary.stateTriangleSet().connect(merge.inFirst())
barricade.stateTriangleSet().connect(merge.inSecond())

fluid.connect(sfi.importParticleSystems())
merge.stateTriangleSet().connect(sfi.inTriangleSet())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
# app.renderWindow().getCamera().setUnitScale(512)
app.mainLoop()
