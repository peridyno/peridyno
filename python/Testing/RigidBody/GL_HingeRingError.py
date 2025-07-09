import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)

rA = dyno.RigidBodyInfo(dyno.Vector3f([0, 0.8, 0]), dyno.Quat1f(0,0,0,1))
rB = dyno.RigidBodyInfo(dyno.Vector3f([0, 0.3, 0]), dyno.Quat1f(0,0,0,1))
rC = dyno.RigidBodyInfo(dyno.Vector3f([-0.26, 0.55, 0]), dyno.Quat1f(0,0,0,1))
rD = dyno.RigidBodyInfo(dyno.Vector3f([0.26, 0.55, 0]), dyno.Quat1f(0,0,0,1))

box1 = box2 = box3 = box4 =  dyno.BoxInfo()
box1.halfLength = dyno.Vector3f([0.03,0.2,0.03])
box1.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([0,0,1]))

box2.halfLength = dyno.Vector3f([0.03,0.2,0.03])
box2.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([0,0,1]))

box3.halfLength = dyno.Vector3f([0.03,0.2,0.03])

box4.halfLength = dyno.Vector3f([0.03,0.2,0.03])

boxActor1 = rigid.addBox(box1, rA)
boxActor2 = rigid.addBox(box2, rB)
boxActor3 = rigid.addBox(box3, rC)
boxActor4 = rigid.addBox(box4, rD)

hingeJoint1 = rigid.createHingeJoint(boxActor1, boxActor3)
hingeJoint1.setAnchorPoint(dyno.Vector3f([-0.26, 0.8, 0]))
hingeJoint1.setAxis(dyno.Vector3f([0,0,1]))
hingeJoint1.setRange(-3.14159265358979323846 * 2 / 3, 3.14159265358979323846 * 2 / 3)
hingeJoint2 = rigid.createHingeJoint(boxActor1, boxActor4)
hingeJoint2.setAnchorPoint(dyno.Vector3f([0.26, 0.8, 0]))
hingeJoint2.setAxis(dyno.Vector3f([0,0,1]))
hingeJoint2.setRange(-3.14159265358979323846 * 2 / 3, 3.14159265358979323846 * 2 / 3)
hingeJoint3 = rigid.createHingeJoint(boxActor2, boxActor3)
hingeJoint3.setAnchorPoint(dyno.Vector3f([-0.26, 0.3, 0]))
hingeJoint3.setAxis(dyno.Vector3f([0,0,1]))
hingeJoint3.setRange(-3.14159265358979323846 * 2 / 3, 3.14159265358979323846 * 2 / 3)
hingeJoint4 = rigid.createHingeJoint(boxActor2, boxActor4)
hingeJoint4.setAnchorPoint(dyno.Vector3f([0.26, 0.3, 0]))
hingeJoint4.setAxis(dyno.Vector3f([0,0,1]))
hingeJoint4.setRange(-3.14159265358979323846 * 2 / 3, 3.14159265358979323846 * 2 / 3)

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
