import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([3, 3, 3]))
scn.setLowerBound(dyno.Vector3f([-3, -3, -3]))

emitter = dyno.PoissonEmitter3f()
scn.addNode(emitter)
emitter.varRotation().setValue(dyno.Vector3f([0, 0, -90]))
emitter.varSamplingDistance().setValue(0.008)
emitter.varEmitterShape().getDataPtr().setCurrentKey(1)
emitter.varWidth().setValue(0.1)
emitter.varHeight().setValue(0.1)
emitter.varVelocityMagnitude().setValue(1.5)
emitter.varLocation().setValue(dyno.Vector3f([0.2, 0.5, 0]))

emitter2 = dyno.PoissonEmitter3f()
scn.addNode(emitter2)
emitter2.varRotation().setValue(dyno.Vector3f([0, 0, 90]))
emitter2.varSamplingDistance().setValue(0.008)
emitter2.varEmitterShape().getDataPtr().setCurrentKey(1)
emitter2.varWidth().setValue(0.1)
emitter2.varHeight().setValue(0.1)
emitter2.varVelocityMagnitude().setValue(1.5)
emitter2.varLocation().setValue(dyno.Vector3f([-0.2, 0.5, -0]))

temp = dyno.DualParticleFluid3f()
fluid = dyno.DualParticleFluid3f(temp.EVirtualParticleSamplingStrategy.SpatiallyAdaptiveStrategy)
scn.addNode(fluid)

emitter.connect(fluid.importParticleEmitters())
emitter2.connect(fluid.importParticleEmitters())

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
vpRender.varPointSize().setValue(0.001)
fluid.graphicsPipeline().pushModule(vpRender)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
