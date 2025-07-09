from os import supports_fd

import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([1.5, 1, 1.5]))
scn.setLowerBound(dyno.Vector3f([-0.5, 0, -0.5]))

cube = dyno.CubeModel3f()
scn.addNode(cube)
cube.varLocation().setValue(dyno.Vector3f([0.5,0.1,0.5]))
cube.varLength().setValue(dyno.Vector3f([0.04,0.04,0.04]))
cube.setVisible(False)

sampler = dyno.ShapeSampler3f()
scn.addNode(sampler)
sampler.varSamplingDistance().setValue(0.005)
sampler.setVisible(False)

cube.connect(sampler.importShape())

initialParticles = dyno.MakeParticleSystem3f()
scn.addNode(initialParticles)
sampler.statePointSet().promoteOutput().connect(initialParticles.inPoints())

emitter = dyno.SquareEmitter3f()
scn.addNode(emitter)
emitter.varLocation().setValue(dyno.Vector3f([0.5,0.5,0.5]))

fluid = dyno.ParticleFluid3f()
scn.addNode(fluid)
initialParticles.connect(fluid.importInitialStates())
emitter.connect(fluid.importParticleEmitters())

cubeBoundary = dyno.CubeModel3f()
scn.addNode(cubeBoundary)
cubeBoundary.varLocation().setValue(dyno.Vector3f([0.5, 0.5,0.5]))
cubeBoundary.varLength().setValue(dyno.Vector3f([1,1,1]))
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

meshRe = dyno.ParticleSkinning3f()
scn.addNode(meshRe)
meshRe.stateGridSpacing().setValue(0.005)
fluid.connect(meshRe.importParticleSystems())

marchingCubes = dyno.MarchingCubes3f()
scn.addNode(marchingCubes)
meshRe.stateLevelSet().connect(marchingCubes.inLevelSet())
marchingCubes.varIsoValue().setValue(-300000)
marchingCubes.varGridSpacing().setValue(0.005)

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.setColor(dyno.Color(0.1,0.1,0.9))
marchingCubes.stateTriangleSet().connect(surfaceRenderer.inTriangleSet())
surfaceRenderer.varAlpha().setValue(0.3)
surfaceRenderer.varMetallic().setValue(0.5)
marchingCubes.graphicsPipeline().pushModule(surfaceRenderer)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
# app.renderWindow().getCamera().setUnitScale(512)
app.mainLoop()
