import PyPeridyno as dyno

scn = dyno.SceneGraph()

cube = dyno.CubeModel3f()
scn.addNode(cube)
cube.setVisible(False)
cube.varLocation().setValue(dyno.Vector3f([0.2, 0.2, 0]))
cube.varLength().setValue(dyno.Vector3f([0.1, 0.1, 0.1]))
cube.varSegments().setValue(dyno.Vector3i([10, 10, 10]))

sampler = dyno.ShapeSampler3f()
scn.addNode(sampler)
sampler.varSamplingDistance().setValue(0.005)
sampler.graphicsPipeline().disable()

cube.connect(sampler.importShape())

initialParticles = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles)

sampler.statePointSet().promoteOuput().connect(initialParticles.inPoints())

elastoplasticBody = dyno.ElastoplasticBody3f()
scn.addNode(elastoplasticBody)
initialParticles.connect(elastoplasticBody.importSolidParticles())

elastoplasticBody.animationPipeline().clear()

integrator = dyno.ParticleIntegrator3f()
elastoplasticBody.stateTimeStep().connect(integrator.inTimeStep())
elastoplasticBody.statePosition().connect(integrator.inPosition())
elastoplasticBody.stateVelocity().connect(integrator.inVelocity())
elastoplasticBody.animationPipeline().pushModule(integrator)

nbrQuery = dyno.NeighborPointQuery3f()
elastoplasticBody.stateHorizon().connect(nbrQuery.inRadius())
elastoplasticBody.statePosition().connect(nbrQuery.inPosition())
elastoplasticBody.animationPipeline().pushModule(nbrQuery)

plasticity = dyno.FractureModule3f()
plasticity.varCohesion().setValue(0.00001)
elastoplasticBody.stateHorizon().connect(plasticity.inHorizon())
elastoplasticBody.stateTimeStep().connect(plasticity.inTimeStep())
elastoplasticBody.statePosition().connect(plasticity.inY())
elastoplasticBody.stateReferencePosition().connect(plasticity.inX())
elastoplasticBody.stateVelocity().connect(plasticity.inVelocity())
# elastoplasticBody.stateBonds().connect(plasticity.inBonds())
nbrQuery.outNeighborIds().connect(plasticity.inNeighborIds())
elastoplasticBody.animationPipeline().pushModule(plasticity)

visModule = dyno.ImplicitViscosity3f()
visModule.varViscosity().setValue(1)
elastoplasticBody.stateTimeStep().connect(visModule.inTimeStep())
elastoplasticBody.stateHorizon().connect(visModule.inSmoothingLength())
elastoplasticBody.statePosition().connect(visModule.inPosition())
elastoplasticBody.stateVelocity().connect(visModule.inVelocity())
nbrQuery.outNeighborIds().connect(visModule.inNeighborIds())
elastoplasticBody.animationPipeline().pushModule(visModule)

cubeBoundary = dyno.CubeModel3f()
scn.addNode(cubeBoundary)
cubeBoundary.varLocation().setValue(dyno.Vector3f([0.5, 1, 0.5]))
cubeBoundary.varLength().setValue(dyno.Vector3f([2, 2, 2]))
cubeBoundary.setVisible(False)

cube2vol = dyno.BasicShapeToVolume3f()
scn.addNode(cube2vol)
cube2vol.varGridSpacing().setValue(0.02)
cube2vol.varInerted().setValue(True)
cubeBoundary.connect(cube2vol.importShape())

container = dyno.VolumeBoundary3f()
scn.addNode(container)
container.varTangentialFriction().setValue(0.95)
cube2vol.connect(container.importVolumes())

elastoplasticBody.connect(container.importParticleSystems())



app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(800, 600, True)
app.mainLoop()
