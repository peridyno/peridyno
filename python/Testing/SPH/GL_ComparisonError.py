import PyPeridyno as dyno

PARTICLE_SPACING = 0.005
SMOOTHINGLEHGTH = PARTICLE_SPACING*2.0
TIME_STEP_SIZE = 0.001

scn = dyno.SceneGraph()
scn.setTotalTime(3)
scn.setGravity(dyno.Vector3f([0, 0, 0]))
scn.setLowerBound(dyno.Vector3f([-2,-2,-2]))
scn.setUpperBound(dyno.Vector3f([2,2,2]))

cube = dyno.CubeModel3f()
scn.addNode(cube)
cube.varLocation().setValue(dyno.Vector3f([0,0,0]))
cube.varScale().setValue(dyno.Vector3f([0.02,0.5,0.5]))

sampler = dyno.ShapeSampler3f()
scn.addNode(sampler)
sampler.varSamplingDistance().setValue(PARTICLE_SPACING)
sampler.setVisible(False)

cube.connect(sampler.importShape())

initialParticles = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles)
initialParticles.varInitialVelocity().setValue(dyno.Vector3f([0,0,0]))

sampler.statePointSet().promoteOutput().connect(initialParticles.inPoints())

sphere2 = dyno.SphereModel3f()
scn.addNode(sphere2)
sphere2.varLocation().setValue(dyno.Vector3f([-0.3,0,0]))
sphere2.varRadius().setValue(0.05)

sampler2 = dyno.ShapeSampler3f()
scn.addNode(sampler2)
sampler2.varSamplingDistance().setValue(PARTICLE_SPACING)
sampler.setVisible(False)

sphere2.connect(sampler2.importShape())

initialParticles2 = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles2)
initialParticles2.varInitialVelocity().setValue(dyno.Vector3f([1,0,0]))

sampler2.statePointSet().promoteOutput().connect(initialParticles2.inPoints())

fluid = dyno.ParticleFluid3f()
scn.addNode(fluid)
fluid.setDt(TIME_STEP_SIZE)
fluid.varReshuffleParticles().setValue(True)
initialParticles.connect(fluid.importInitialStates())
initialParticles2.connect(fluid.importInitialStates())

fluid.animationPipeline().clear()
smoothingLength = fluid.animationPipeline().createModules()
smoothingLength.varValue().setValue(SMOOTHINGLEHGTH)

samplingDistance = fluid.animationPipeline().createModules()
samplingDistance.varValue().setValue(PARTICLE_SPACING)

integrator = dyno.ParticleIntegrator3f()
fluid.stateTimeStep().connect(integrator.inTimeStep())
fluid.statePosition().connect(integrator.inPosition())
fluid.stateVelocity().connect(integrator.inVelocity())
fluid.animationPipeline().pushModule(integrator)

nbrQuery = dyno.NeighborPointQuery3f()
smoothingLength.outFloating().connect(nbrQuery.inRadius())
fluid.statePosition().connect(nbrQuery.inPosition())
fluid.animationPipeline().pushModule(nbrQuery)

density = dyno.ImplicitISPH3f()
density.varIterationNumber().setValue(10)
smoothingLength.outFloating().connect(density.inSmoothingLength())
samplingDistance.outFloating().connect(density.inSamplingDistance())
fluid.stateTimeStep().connect(density.inTimeStep())
fluid.statePosition().connect(density.inPosition())
fluid.stateVelocity().connect(density.inVelocity())
nbrQuery.outNeighborIds().connect(density.inNeighborIds())
fluid.animationPipeline().pushModule(density)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
# app.renderWindow().getCamera().setUnitScale(512)
app.mainLoop()
