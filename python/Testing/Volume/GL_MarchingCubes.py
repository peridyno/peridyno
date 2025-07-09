from os import supports_fd

import PyPeridyno as dyno

scn = dyno.SceneGraph()

loader = dyno.VolumeLoader3f()
scn.addNode(loader)
loader.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "bowl/bowl.sdf"))

clipper = dyno.VolumeClipper3f()
scn.addNode(clipper)
loader.stateLevelSet().connect(clipper.inLevelSet())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
# app.renderWindow().getCamera().setUnitScale(512)
app.mainLoop()
