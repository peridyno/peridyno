import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)
rA = rB = rC = rD = dyno.RigidBodyInfo()
box1 = box2 = box3 =  dyno.BoxInfo()

rA.position = dyno.Vector3f([0,0.5,0])
box1.halfLength = dyno.Vector3f([1,0.05,1])

rB.position = dyno.Vector3f([0,0.2,0.8])
box2.halfLength = dyno.Vector3f([0.7,0.1,0.1])

rC.position = dyno.Vector3f([0,0.2,-0.8])
box3.halfLength = dyno.Vector3f([0.7,0.1,0.1])

bodyActor = rigid.addBox(box1, rA)
frontActor = rigid.addBox(box2, rB)
rearActor = rigid.addBox(box3, rC)

sphere = dyno.SphereInfo()
sphere.radius = 0.1

rA.position = dyno.Vector3f([0.9,0.1,0.8])
rB.position = dyno.Vector3f([-0.9,0.1,0.8])
rC.position = dyno.Vector3f([0.9,0.1,-0.8])
rD.position = dyno.Vector3f([-0.9,0.1,-0.8])

frontLeftTire = rigid.addSphere(sphere, rA)
frontRightTire = rigid.addSphere(sphere, rB)
rearLeftTire = rigid.addSphere(sphere, rC)
rearRightTire = rigid.addSphere(sphere, rD)

joint1 = rigid.createHingeJoint(frontLeftTire, frontActor)
joint1.setAnchorPoint(frontLeftTire.center)
joint1.setAxis(dyno.Vector3f([1,0,0]))
joint1.setMoter(30)

joint2 = rigid.createHingeJoint(frontRightTire, frontActor)
joint2.setAnchorPoint(frontRightTire.center)
joint2.setAxis(dyno.Vector3f([1,0,0]))
joint2.setMoter(30)

joint3 = rigid.createHingeJoint(rearLeftTire, rearActor)
joint3.setAnchorPoint(rearLeftTire.center)
joint3.setAxis(dyno.Vector3f([1,0,0]))
joint3.setMoter(30)

joint4 = rigid.createHingeJoint(rearRightTire, rearActor)
joint4.setAnchorPoint(rearRightTire.center)
joint4.setAxis(dyno.Vector3f([1,0,0]))
joint4.setMoter(30)

joint5 = rigid.createFixedJoint(rearActor, bodyActor)
joint5.setAnchorPoint(rearActor.center)

joint6 = rigid.createHingeJoint(frontActor, bodyActor)
joint6.setAnchorPoint(frontActor.center)
joint6.setAxis(dyno.Vector3f([0,1,0]))
joint6.setRange(M_PI / 12, M_PI / 12)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(0.204, 0.424, 0.612))
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

# contactPointMapper = dyno.ContactsToPointSet3f()
# elementQuery.outContacts().connect(contactPointMapper.inContacts())
# rigid.graphicsPipeline().pushModule(contactPointMapper)
#
# pointRender = dyno.GLPointVisualModule()
# pointRender.setColor(dyno.Color(1, 0, 0))
# pointRender.varPointSize().setValue(0.003)
# contactPointMapper.outPointSet().connect(pointRender.inPointSet())
# rigid.graphicsPipeline().pushModule(pointRender)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()
