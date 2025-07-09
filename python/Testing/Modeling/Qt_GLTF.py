import PyPeridyno as dyno

scn = dyno.SceneGraph()

gltf = dyno.GltfLoader3f()
scn.addNode(gltf)

gltf.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/JeepGltf/jeep.gltf"))


app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
