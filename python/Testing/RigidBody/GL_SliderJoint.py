import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)

rigidBody = dyno.RigidBodyInfo()
rigidBody.friction = 0.01
box1 = box2 = dyno.BoxInfo()

box1.halfLength = dyno.Vector3f([0.09, 0.1, 0.1])

rigidBody.position = dyno.Vector3f([0, 0.5,0])

boxActor1 = rigid.addBox(box1, rigidBody, 1000)

box2.halfLength = dyno.Vector3f([0.1, 0.4,0.2])

rigidBody.position = dyno.Vector3f([0.2, 0.5, 0])
rigidBody.angularVelocity = dyno.Vector3f([1,0,0])
rigidBody.motionType = dyno.BodyType.Kinematic

boxActor2 = rigid.addBox(box2, rigidBody)

sliderJoint = rigid.createSliderJoint(boxActor1, boxActor2)

sliderJoint.setAnchorPoint((boxActor1.center + boxActor2.center) / 2)

sliderJoint.setAxis(dyno.Vector3f([0,1,0]))
sliderJoint.setRange(-0.2, 0.2)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(1,1,0))
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
