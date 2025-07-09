from os import supports_fd

import PyPeridyno as dyno

scn = dyno.SceneGraph()

mesh = dyno.TextureMeshLoader3f()
scn.addNode(mesh)
mesh.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "obj/standard/cube.obj"))
mesh.varScale().setValue(dyno.Vector3f([0.3,0.3,0.3]))
mesh.varLocation().setValue(dyno.Vector3f([-1.5,0.3,0]))

mesh1 = dyno.TextureMeshLoader3f()
scn.addNode(mesh1)
mesh1.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "obj/moon/Moon_Normal.obj"))
mesh1.varScale().setValue(dyno.Vector3f([0.005,0.005,0.005]))
mesh1.varLocation().setValue(dyno.Vector3f([0.5,0.3,0.5]))


app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
# app.renderWindow().getCamera().setUnitScale(512)
app.mainLoop()
