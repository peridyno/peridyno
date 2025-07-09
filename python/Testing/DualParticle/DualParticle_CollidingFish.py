import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([3, 3, 3]))
scn.setLowerBound(dyno.Vector3f([-3, -3, -3]))
scn.setGravity(dyno.Vector3f([0, 0, 0]))

ptsLoader = dyno.PointsLoader3f()
scn.addNode(ptsLoader)
ptsLoader.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "fish/FishPoints.obj"))
ptsLoader.varRotation().setValue(dyno.Vector3f([0, 0, 3.1415926]))
ptsLoader.varLocation().setValue(dyno.Vector3f([0, 0, 0.23]))
initialParticles = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles)
initialParticles.varInitialVelocity().setValue(dyno.Vector3f([0, 0, -1.5]))
ptsLoader.outPointSet().promoteOuput().connect(initialParticles.inPoints())

ptsLoader2 = dyno.PointsLoader3f()
scn.addNode(ptsLoader2)
ptsLoader2.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "fish/FishPoints.obj"))
ptsLoader2.varRotation().setValue(dyno.Vector3f([0, 0, 0]))
ptsLoader2.varLocation().setValue(dyno.Vector3f([0, 0, -0.23]))
initialParticles2 = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles2)
initialParticles2.varInitialVelocity().setValue(dyno.Vector3f([0, 0, 1.5]))
ptsLoader2.outPointSet().promoteOuput().connect(initialParticles2.inPoints())

fluid = dyno.DualParticleFluid3f()
scn.addNode(fluid)
fluid.varReshuffleParticles().setValue(True)
initialParticles.connect(fluid.importInitialStates())
initialParticles2.connect(fluid.importInitialStates())

calculateNorm = dyno.CalculateNorm3f()
fluid.stateVelocity().connect(calculateNorm.inVec())
fluid.graphicsPipeline().pushModule(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.varMax().setValue(5.0)
calculateNorm.outNorm().connect(colorMapper.inScalar())
fluid.graphicsPipeline().pushModule(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.setColor(dyno.Color(1, 0, 0))
ptRender.varPointSize().setValue(0.0035)
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
fluid.stateVirtualPointSet().connect(vpRender.inPointSet())
vpRender.varPointSize().setValue(0.0005)
fluid.graphicsPipeline().pushModule(vpRender)


app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
