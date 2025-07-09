import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)

box =  dyno.BoxInfo()
rA = dyno.RigidBodyInfo()
rA.bodyId = 1
rA.linearVelocity = dyno.Vector3f([1,0,0])
box.center = dyno.Vector3f([0,0,0])
box.halfLength = dyno.Vector3f([0.05,0.05,0.05])
oldBoxActor = rigid.addBox(box, rA)

for i in range(100):
    rB = dyno.RigidBodyInfo()
    rB.position = rA.position + dyno.Vector3f([0, 0.12, 0])
    rB.linearVelocity = dyno.Vector3f([0,0,0])

    newBoxActor = rigid.addBox(box, rB)

    ballAndSockerJoint = rigid.createBallAndSocketJoint(oldBoxActor, newBoxActor)
    ballAndSockerJoint.setAnchorPoint((rA.position + rB.position) / 2)

    rA = rB
    oldBoxActor = newBoxActor

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(0.3,0.5,0.9))
sRender.setAlpha(0.8)
sRender.setRoughness(0.7)
sRender.setMetallic(3)
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
wireRender.setColor(dyno.Color(0, 0, 1))
contactMapper.outEdgeSet().connect(wireRender.inEdgeSet())
rigid.graphicsPipeline().pushModule(wireRender)

contactPointMapper = dyno.ContactsToPointSet3f()
elementQuery.outContacts().connect(contactPointMapper.inContacts())
rigid.graphicsPipeline().pushModule(contactPointMapper)

pointRender = dyno.GLPointVisualModule()
pointRender.setColor(dyno.Color(1, 0, 0))
pointRender.varPointSize().setValue(0.003)
contactPointMapper.outPointSet().connect(pointRender.inPointSet())
rigid.graphicsPipeline().pushModule(pointRender)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()
