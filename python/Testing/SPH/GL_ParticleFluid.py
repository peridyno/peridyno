import PyPeridyno as dyno

scn = dyno.SceneGraph()

scn.setUpperBound(dyno.Vector3f([1.5, 1, 1.5]))
scn.setLowerBound(dyno.Vector3f([-0.5, 0, -0.5]))

cube = dyno.CubeModel3f()
scn.addNode(cube)
cube.varLocation().setValue(dyno.Vector3f([0.6, 0.5, 0.5]))
cube.varLength().setValue(dyno.Vector3f([0.5,0.5,0.5]))
cube.graphicsPipeline().disable()

sampler = dyno.ShapeSampler3f()
scn.addNode(sampler)
sampler.varSamplingDistance().setValue(0.005)
sampler.graphicsPipeline().disable()

cube.connect(sampler.importShape())

initialParticles = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles)

sampler.statePointSet().promoteOutput().connect(initialParticles.inPoints())

fluid = dyno.ParticleFluid3f()
scn.addNode(fluid)
fluid.varReshuffleParticles().setValue(True)
initialParticles.connect(fluid.importInitialStates())

volLoader = dyno.VolumeLoader3f()
scn.addNode(volLoader)
volLoader.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "bowl/bowl.sdf"))

volBoundary = dyno.VolumeBoundary3f()
scn.addNode(volBoundary)
volLoader.connect(volBoundary.importVolumes())

fluid.connect(volBoundary.importParticleSystems())

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
cube2vol.connect(volBoundary.importVolumes())

staticMesh = dyno.StaticMeshLoader3f()
scn.addNode(staticMesh)
staticMesh.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() +"bowl/bowl.obj" ))

fluid.graphicsPipeline().clear()

calculateNorm = dyno.CalculateNorm3f()
fluid.stateVelocity().connect(calculateNorm.inVec())
fluid.graphicsPipeline().pushModule(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.varMax().setValue(5)
calculateNorm.outNorm().connect(colorMapper.inScalar())
fluid.graphicsPipeline().pushModule(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.setColor(dyno.Color(1,0,0))
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
app.initialize(800, 600, True)
app.mainLoop()
