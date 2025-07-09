import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)

box =  dyno.BoxInfo()
box.center = dyno.Vector3f([0,0,0])
box.halfLength = dyno.Vector3f([0.02, 0.02,0.02])

rbA = dyno.RigidBodyInfo()
rbB = dyno.RigidBodyInfo()
rbA.position = dyno.Vector3f([0,0.1,0])
rbA.linearVelocity = dyno.Vector3f([1, 0, 1])

oldBoxActor = rigid.addBox(box, rbA)

rbA.linearVelocity = dyno.Vector3f([0,0,0])

for i in range(1, 300):
    rbB.position = rbA.position + dyno.Vector3f([0, 0.05, 0])
    rbB.angle = dyno.Quat1f(3.14159265358979323846 / 3 * i, dyno.Vector3f([0,1,0]))
    newBoxActor = rigid.addBox(box, rbB)

    fixedJoint = rigid.createFixedJoint(oldBoxActor, newBoxActor)
    fixedJoint.setAnchorPoint((rbA.position + rbB.position) / 2)

    rbA = rbB
    oldBoxActor = newBoxActor

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(1, 1, 0))
sRender.setAlpha(1.0)
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
