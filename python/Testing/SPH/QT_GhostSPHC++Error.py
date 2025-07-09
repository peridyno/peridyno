import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([10.5,5,10.5]))
scn.setLowerBound(dyno.Vector3f([-10.5,5,-10.5]))

obj1 = dyno.ObjLoader3f()
scn.addNode(obj1)
obj1.varScale().setValue(dyno.Vector3f([0.3,0.3,0.3]))
obj1.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "plane/plane_lowRes.obj"))
obj1.varLocation().setValue(dyno.Vector3f([0,0,0]))
SurfaceModule1 = obj1.graphicsPipeline().findFirstModuleSurface()
SurfaceModule1.setColor(dyno.Color(0.2, 0.2, 0.2))
SurfaceModule1.setMetallic(1)
SurfaceModule1.setRoughness(0.8)

pointset1 = dyno.ParticleRelaxtionOnMesh3f()
scn.addNode(pointset1)
pointset1.varSamplingDistance().setValue(0.005)
pointset1.varThickness().setValue(0.045)
obj1.outTriangleSet().connect(pointset1.inTriangleSet())
pointset1.graphicsPipeline().clear()

ghost2 = dyno.MakeGhostParticles3f()
scn.addNode(ghost2)
pointset1.statePointSet().connect(ghost2.inPoints())
pointset1.statePointNormal().connect(ghost2.stateNormal())

cube = dyno.CubeModel3f()
scn.addNode(cube)
cube.varLocation().setValue(dyno.Vector3f([0,0.3,0]))
cube.varLength().setValue(dyno.Vector3f([0.2,0.2,0.2]))
cube.setVisible(False)

sampler = dyno.ShapeSampler3f()
scn.addNode(sampler)
sampler.varSamplingDistance().setValue(0.005)
sampler.setVisible(False)

cube.connect(sampler.importShape())

fluidParticles = dyno.MakeParticleSystem3f()
scn.addNode(fluidParticles)

sampler.statePointSet().promoteOutput().connect(fluidParticles.inPoints())

incompressibleFluid = dyno.GhostFluid3f()
scn.addNode(incompressibleFluid)
fluidParticles.connect(incompressibleFluid.importInitialStates())
ghost2.connect(incompressibleFluid.importBoundaryParticles())


ptRender = dyno.GLPointVisualModule()
ptRender.setColor(dyno.Color(0.6,0.5,0.2))
ptRender.setColorMapMode(ptRender.ColorMapMode.PER_VERTEX_SHADER)
pointset1.statePointSet().connect(ptRender.inPointSet())
pointset1.graphicsPipeline().pushModule(ptRender)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
