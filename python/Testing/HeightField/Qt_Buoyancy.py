import PyPeridyno as dyno

scn = dyno.SceneGraph()

ocean = dyno.Ocean3f()
ocean.varExtentX().setValue(2)
ocean.varExtentZ().setValue(2)
scn.addNode(ocean)

patch = dyno.OceanPatch3f()
scn.addNode(patch)
patch.varWindType().setValue(5)
patch.varPatchSize().setValue(128)
patch.connect(ocean.importOceanPatch())
patch.varResolution().setValue(512)

wake = dyno.Wake3f()
scn.addNode(wake)
wake.varWaterLevel().setValue(4)
wake.varLength().setValue(128)
wake.varMagnitude().setValue(0.2)
wake.connect(ocean.importCapillaryWaves())

mapper = dyno.HeightFieldToTriangleSet3f()

ocean.stateHeightField().connect(mapper.inHeightField())
ocean.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(0,0.2,1))
sRender.varUseVertexNormal().setValue(False)
sRender.varAlpha().setValue(0.6)
mapper.outTriangleSet().connect(sRender.inTriangleSet())
ocean.graphicsPipeline().pushModule(sRender)

boat = dyno.Vessel3f()
scn.addNode(boat)
boat.varDensity().setValue(150)
boat.varBarycenterOffset().setValue(dyno.Vector3f([0,0,-0.5]))
boat.stateVelocity().setValue(dyno.Vector3f([0,0,0]))
boat.varEnvelopeName().setValue(dyno.FilePath(dyno.getAssetPath() + "obj/boat_boundary.obj"))
boat.varTextureMeshName().setValue(dyno.FilePath(dyno.getAssetPath() + "gltf/SailBoat/SailBoat.gltf"))

steer = dyno.Steer3f()
boat.stateVelocity().connect(steer.inVelocity())
boat.stateAngularVelocity().connect(steer.inAngularVelocity())
boat.stateQuaternion().connect(steer.inQuaternion())
boat.animationPipeline().pushModule(steer)

coupling = dyno.RigidWaterCoupling3f()
scn.addNode(coupling)
boat.connect(wake.importVessel())
boat.connect(coupling.importVessels())
ocean.connect(coupling.importOcean())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
