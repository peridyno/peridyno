import PyPeridyno as dyno
from PyPeridyno import ObjLoader3f

scn = dyno.SceneGraph()
scn.setTotalTime(3.0)
scn.setGravity(dyno.Vector3f([0, -9.8, 0]))
scn.setUpperBound(dyno.Vector3f([0.5, 1, 4]))
scn.setLowerBound(dyno.Vector3f([-0.5, 0, 4]))

velocity = dyno.Vector3f([0, 0, 6])
color = dyno.Color(1, 1, 1)

LocationBody = dyno.Vector3f([0, 0.01, -1])

anglurVel = dyno.Vector3f([100, 0, 0])
scale = dyno.Vector3f([0.4, 0.4, 0.4])

ObjJeep = dyno.ObjLoader3f()
scn.addNode(ObjJeep)
ObjJeep.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/jeepLow.obj"))
ObjJeep.varScale().setValue(scale)
ObjJeep.varLocation().setValue(LocationBody)
ObjJeep.varVelocity().setValue(velocity)
glJeep = ObjJeep.graphicsPipeline().findFirstModuleSurface()
glJeep.setColor(color)

wheelPath = ["Jeep/Wheel_R.obj","Jeep/Wheel_R.obj","Jeep/Wheel_L.obj","Jeep/Wheel_R.obj"]
wheelSet = []

wheelLocation = []
wheelLocation.append(dyno.Vector3f([0.17, 0.1, 0.36]) + LocationBody)
wheelLocation.append(dyno.Vector3f([0.17, 0.1, -0.3]) + LocationBody)
wheelLocation.append(dyno.Vector3f([-0.17, 0.1, 0.36]) + LocationBody)
wheelLocation.append(dyno.Vector3f([-0.17, 0.1, -0.3]) + LocationBody)

for i in range(4):
    ObjWheel = dyno.ObjLoader3f()
    scn.addNode(ObjWheel)

    ObjWheel.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + wheelPath[i]))

    ObjWheel.varScale().setValue(scale)
    ObjWheel.varLocation().setValue(wheelLocation[i])
    ObjWheel.varCenter().setValue(wheelLocation[i])

    ObjWheel.varVelocity().setValue(velocity)
    ObjWheel.varAngularVelocity().setValue(anglurVel)

    wheelSet.append(ObjWheel)

# Import Road
ObjRoad = dyno.ObjLoader3f()
scn.addNode(ObjRoad)
ObjRoad.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/Road/Road.obj"))
ObjRoad.varScale().setValue(dyno.Vector3f([0.04, 0.04, 0.04]))
ObjRoad.varLocation().setValue(dyno.Vector3f([0, 0, 0.5]))
glRoad = ObjRoad.graphicsPipeline().findFirstModuleSurface()
glRoad.setColor(color)

# *************************************** Merge Model ***************************************//
# MergeWheel
mergeWheel = dyno.Merge3f()
scn.addNode(mergeWheel)
mergeWheel.varUpdateMode().setCurrentKey(1)

wheelSet[0].outTriangleSet().connect(mergeWheel.inTriangleSet01())
wheelSet[1].outTriangleSet().connect(mergeWheel.inTriangleSet02())
wheelSet[2].outTriangleSet().connect(mergeWheel.inTriangleSet03())
wheelSet[3].outTriangleSet().connect(mergeWheel.inTriangleSet04())

# MergeRoad
mergeRoad = dyno.Merge3f()
scn.addNode(mergeRoad)
mergeRoad.varUpdateMode().setCurrentKey(1)
mergeWheel.stateTriangleSet().promoteOutput().connect(mergeRoad.inTriangleSet01())
ObjRoad.outTriangleSet().connect(mergeRoad.inTriangleSet03())

# Obj boundary
ObjBoundary = dyno.ObjLoader3f()
scn.addNode(ObjBoundary)
ObjBoundary.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/Road/boundary.obj"))
ObjBoundary.varScale().setValue(dyno.Vector3f([0.04, 0.04, 0.04]))
ObjBoundary.varLocation().setValue(dyno.Vector3f([0, 0, 0.5]))
glBoundary = ObjBoundary.graphicsPipeline().findFirstModuleSurface()
glBoundary.setColor(color)

ObjBoundary.outTriangleSet().connect(mergeRoad.inTriangleSet02())
ObjBoundary.graphicsPipeline().disable()
ObjJeep.outTriangleSet().connect(mergeRoad.inTriangleSet04())

# SetVisible
mergeRoad.graphicsPipeline().disable()

# *************************************** Cube Sample ***************************************
# Cube
cube = dyno.CubeModel3f()
scn.addNode(cube)
cube.varLocation().setValue(dyno.Vector3f([0, 0.025, 0.4]))
cube.varLength().setValue(dyno.Vector3f([0.35, 0.02, 3]))
cube.varScale().setValue(dyno.Vector3f([2, 1, 1]))
cube.graphicsPipeline().disable()

cubeSampler = dyno.ShapeSampler3f()
scn.addNode(cubeSampler)
cubeSampler.varSamplingDistance().setValue(0.005)
cube.connect(cubeSampler.importShape())
cubeSampler.graphicsPipeline().disable()

# MakeParticleSystem
particleSystem = dyno.MakeParticleSystem3f()
scn.addNode(particleSystem)
cubeSampler.statePointSet().promoteOutput().connect(particleSystem.inPoints())

# *************************************** Fluid ***************************************//
# Particle fluid node
fluid = dyno.ParticleFluid3f()
scn.addNode(fluid)
particleSystem.connect(fluid.importInitialStates())

visualizer = dyno.GLPointVisualNode3f()
scn.addNode(visualizer)
ptrender = visualizer.graphicsPipeline().findFirstModulePoint()
ptrender.varPointSize().setValue(0.001)

fluid.statePointSet().promoteOutput().connect(visualizer.inPoints())
fluid.stateVelocity().promoteOutput().connect(visualizer.inVector())

# SemiAnalyticalSFINode
meshBoundary = dyno.TriangularMeshBoundary3f()
scn.addNode(meshBoundary)
fluid.connect(meshBoundary.importParticleSystems())

mergeRoad.stateTriangleSet().promoteOutput().connect(meshBoundary.inTriangleSet())

# Create a boundary
cubeBoundary = dyno.CubeModel3f()
scn.addNode(cubeBoundary)
cubeBoundary.varLocation().setValue(dyno.Vector3f([0, 1, 0.75]))
cubeBoundary.varLength().setValue(dyno.Vector3f([2, 2, 4.5]))
cubeBoundary.setVisible(False)

cube2Vol = dyno.BasicShapeToVolume3f()
scn.addNode(cube2Vol)
cube2Vol.varGridSpacing().setValue(0.02)
cube2Vol.varInerted().setValue(True)
cubeBoundary.connect(cube2Vol.importShape())

container = dyno.VolumeBoundary3f()
scn.addNode(container)
cube2Vol.connect(container.importVolumes())

fluid.connect(container.importParticleSystems())

# first Module
colormapping = visualizer.graphicsPipeline().findFirstModuleColorMapping()
colormapping.varMax().setValue(1.5)

LocationRoad = dyno.Vector3f([0,0,0.5])
ScaleRoad = dyno.Vector3f([0.04,0.04,0.04])

ObjRoad_1 = ObjLoader3f()
scn.addNode(ObjRoad_1)
ObjRoad_1.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/Road/obj1.obj"))
ObjRoad_1.varScale().setValue(ScaleRoad)
ObjRoad_1.varLocation().setValue(LocationRoad)
glRoad_1 = ObjRoad_1.graphicsPipeline().findFirstModuleSurface()
glRoad_1.setColor(dyno.Color(1,1,1))

ObjRoadWall = dyno.ObjLoader3f()
scn.addNode(ObjRoadWall)
ObjRoadWall.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/Road/objWall.obj"))
ObjRoadWall.varScale().setValue(ScaleRoad)
ObjRoadWall.varLocation().setValue(LocationRoad)
glRoadWall = ObjRoadWall.graphicsPipeline().findFirstModuleSurface()
glRoadWall.setColor(dyno.Color(1,1,1))

ObjRoadDoor = dyno.ObjLoader3f()
scn.addNode(ObjRoadDoor)
ObjRoadDoor.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/Road/objDoor.obj"))
ObjRoadDoor.varScale().setValue(ScaleRoad)
ObjRoadDoor.varLocation().setValue(LocationRoad)
glRoadDoor = ObjRoadDoor.graphicsPipeline().findFirstModuleSurface()
glRoadDoor.setColor(dyno.Color(0.5,0.5,0.5))
glRoadDoor.setRoughness(0.5)
glRoadDoor.setMetallic(1)

ObjRoadLogo = dyno.ObjLoader3f()
scn.addNode(ObjRoadLogo)
ObjRoadLogo.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/Road/objLogo.obj"))
ObjRoadLogo.varScale().setValue(ScaleRoad)
ObjRoadLogo.varLocation().setValue(LocationRoad)
glRoadLogo = ObjRoadLogo.graphicsPipeline().findFirstModuleSurface()
glRoadLogo.setColor(dyno.Color(0,0.2,1))

ObjRoadText = dyno.ObjLoader3f()
scn.addNode(ObjRoadText)
ObjRoadText.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/Road/objPeridyno.obj"))
ObjRoadText.varScale().setValue(ScaleRoad)
ObjRoadText.varLocation().setValue(LocationRoad)
glRoadText = ObjRoadText.graphicsPipeline().findFirstModuleSurface()
glRoadText.setColor(dyno.Color(0.5,0.5,0.5))

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1366, 768, True)
app.mainLoop()
