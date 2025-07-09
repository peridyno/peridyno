import PyPeridyno as dyno

scn = dyno.SceneGraph()

emitter = dyno.SquareEmitter3f()
scn.addNode(emitter)
emitter.varLocation().setValue(dyno.Vector3f([0.5, 0.5, 0.5]))

fluid = dyno.ParticleFluid3f()
scn.addNode(fluid)
emitter.connect(fluid.importParticleEmitters())

cubeBoundary = dyno.CubeModel3f()
scn.addNode(cubeBoundary)
cubeBoundary.varLocation().setValue(dyno.Vector3f([0.5,0.5,0.5]))
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

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(800, 600, True)
app.mainLoop()
