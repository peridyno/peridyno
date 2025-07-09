import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)

dim = 5
h = 0.1

rigidBody = dyno.RigidBodyInfo()
rigidBody.linearVelocity = dyno.Vector3f([0.0, 0, 0])
box = dyno.BoxInfo()
box.halfLength = dyno.Vector3f([h, h, h])
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            rigidBody.position = dyno.Vector3f([2 * i * h - h * dim, h + 2.01 * j * h, 2 * k * h - h * dim])
            boxAt = rigid.addBox(box, rigidBody)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(0.36078, 0.67451, 0.93333))
sRender.setAlpha(1)
mapper.outTriangleSet().connect(sRender.inTriangleSet())
rigid.graphicsPipeline().pushModule(sRender)

elementQuery = dyno.NeighborElementQuery3f()
rigid.stateTopology().connect(elementQuery.inDiscreteElements())
rigid.stateCollisionMask().connect(elementQuery.inCollisionMask())
rigid.graphicsPipeline().pushModule(elementQuery)

contactMapper = dyno.ContactsToEdgeSet3f()
elementQuery.outContacts().connect(contactMapper.inContacts())
contactMapper.varScale().setValue(0.02)
rigid.graphicsPipeline().pushModule(contactMapper)

wireRender = dyno.GLWireframeVisualModule()
wireRender.setColor(dyno.Color(0, 0, 0))
mapper.outTriangleSet().connect(wireRender.inEdgeSet())
rigid.graphicsPipeline().pushModule(wireRender)



app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
