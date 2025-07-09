import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([1.5, 1, 1.5]))
scn.setLowerBound(dyno.Vector3f([-0.5, 0, -0.5]))

ptsLoader = dyno.PointsLoader3f()
scn.addNode(ptsLoader)
ptsLoader.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "fish/FishPoints.obj"))
ptsLoader.varRotation().setValue(dyno.Vector3f([0, 0, 3.1415926]))
ptsLoader.varLocation().setValue(dyno.Vector3f([0, 0.15, 0.23]))
initialParticles = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles)
ptsLoader.outPointSet().promoteOutput().connect(initialParticles.inPoints())

fluid = dyno.ParticleFluid3f()
scn.addNode(fluid)
fluid.varReshuffleParticles().setValue(True)
initialParticles.connect(fluid.importInitialStates())

fluid.animationPipeline().clear()

smoothingLength = dyno.FloatingNumber3f()
fluid.animationPipeline().pushModule(smoothingLength)
smoothingLength.varValue().setValue(0.0125)

integrator = dyno.ParticleIntegrator3f()
fluid.stateTimeStep().connect(integrator.inTimeStep())
fluid.statePosition().connect(integrator.inPosition())
fluid.stateVelocity().connect(integrator.inVelocity())
fluid.animationPipeline().pushModule(integrator)

nbrQuery = dyno.NeighborPointQuery3f()
smoothingLength.outFloating().connect(nbrQuery.inRadius())
fluid.statePosition().connect(nbrQuery.inPosition())
fluid.animationPipeline().pushModule(nbrQuery)

simple = dyno.SimpleVelocityConstraint3f()
simple.varViscosity().setValue(500)
simple.varSimpleIterationEnable().setValue(False)
fluid.stateTimeStep().connect(simple.inTimeStep())
smoothingLength.outFloating().connect(simple.inSmoothingLength())
fluid.statePosition().connect(simple.inPosition())
fluid.stateVelocity().connect(simple.inVelocity())

simple.inSamplingDistance().setValue(0.005)
nbrQuery.outNeighborIds().connect(simple.inNeighborIds())
fluid.animationPipeline().pushModule(simple)

cubeBoundary = dyno.CubeModel3f()
scn.addNode(cubeBoundary)
cubeBoundary.varLocation().setValue(dyno.Vector3f([0.5,1,0.5]))
cubeBoundary.varLength().setValue(dyno.Vector3f([2,2,2]))
cubeBoundary.setVisible(False)

cube2vol = dyno.BasicShapeToVolume3f()
scn.addNode(cube2vol)
cube2vol.varGridSpacing().setValue(0.02)
cube2vol.varInerted().setValue(True)
cubeBoundary.connect(cube2vol.importShape())

container = dyno.VolumeBoundary3f()
scn.addNode(container)
cube2vol.connect(container.importVolumes())
fluid.connect(container.importParticleSystems())

calculateNorm = dyno.CalculateNorm3f()
fluid.stateVelocity().connect(calculateNorm.inVec())
fluid.graphicsPipeline().pushModule(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.varMax().setValue(5)
calculateNorm.outNorm().connect(colorMapper.inScalar())
fluid.graphicsPipeline().pushModule(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.setColor(dyno.Color(1, 0, 0))
ptRender.setColorMapMode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

fluid.statePointSet().connect(ptRender.inPointSet())
colorMapper.outColor().connect(ptRender.inColor())

fluid.graphicsPipeline().pushModule(ptRender)

colorBar = dyno.ImColorbar3f()
colorBar.varMax().setValue(5)
colorBar.varFieldName().setValue("Velocity")
calculateNorm.outNorm().connect(colorBar.inScalar())
fluid.graphicsPipeline().pushModule(colorBar)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
# app.renderWindow().getCamera().setUnitScale(512)
app.mainLoop()
