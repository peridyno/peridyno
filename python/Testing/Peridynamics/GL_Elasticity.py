import PyPeridyno as dyno

scn = dyno.SceneGraph()

cube = dyno.CubeModel3f()
scn.addNode(cube)
cube.varLocation().setValue(dyno.Vector3f([0.6, 0.2, 0.5]))
cube.varLength().setValue(dyno.Vector3f([0.1, 0.1, 0.1]))
cube.graphicsPipeline().disable()

sampler = dyno.ShapeSampler3f()
scn.addNode(sampler)
sampler.varSamplingDistance().setValue(0.005)
sampler.graphicsPipeline().disable()

cube.connect(sampler.importShape())

initialParticles = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles)

sampler.statePointSet().promoteOuput().connect(initialParticles.inPoints())

bunny = dyno.ElasticBody3f()
scn.addNode(bunny)

initialParticles.connect(bunny.importSolidParticles())

cubeBoundary = dyno.CubeModel3f()
scn.addNode(cubeBoundary)
cubeBoundary.varLocation().setValue(dyno.Vector3f([0.5, 0.5, 0.5]))
cubeBoundary.varLength().setValue(dyno.Vector3f([1, 1, 1]))
cubeBoundary.setVisible(False)

cube2vol = dyno.BasicShapeToVolume3f()
scn.addNode(cube2vol)
cube2vol.varGridSpacing().setValue(0.02)
cube2vol.varInerted().setValue(True)
cubeBoundary.connect(cube2vol.importShape())

container = dyno.VolumeBoundary3f()
scn.addNode(container)
cube2vol.connect(container.importVolumes())

bunny.connect(container.importParticleSystems())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
