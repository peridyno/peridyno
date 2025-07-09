import PyPeridyno as dyno

scn = dyno.SceneGraph()

root = dyno.OceanPatch3f()
scn.addNode(root)
root.varWindType().setValue(8)

mapper = dyno.HeightFieldToTriangleSet3f()
root.stateHeightField().connect(mapper.inHeightField())
root.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(0, 0.2, 1.0))
sRender.varUseVertexNormal().connect(sRender.inTriangleSet())
mapper.outTriangleSet().connect(sRender.inTriangleSet())
root.graphicsPipeline().pushModule(sRender)



app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
