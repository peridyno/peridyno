import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)

rigidBody = dyno.RigidBodyInfo()
scale = 2.5
sphere = dyno.SphereInfo()
sphere.center = dyno.Vector3f([scale, scale, scale]) * dyno.Vector3f([-4.6,20,0.5])
sphere.radius = scale * 2.5

newbox = oldbox = dyno.BoxInfo()
oldbox.center = dyno.Vector3f([scale, scale, scale]) * dyno.Vector3f([-2.0,20,0.5])
oldbox.halfLength = dyno.Vector3f([scale, scale, scale]) * dyno.Vector3f([0.05,0.09,0.02])
oldbox.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([0,0,0]))
oldBoxActor = rigid.addBox(oldbox, rigidBody)
rigidBody.linearVelocity = dyno.Vector3f([0,0,0])

for i in range(20):
    newbox.center = oldbox.center + dyno.Vector3f([scale, scale, scale]) * dyno.Vector3f([0.2,0,0])
    newbox.halfLength = oldbox.halfLength
    newbox.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([0,0,1]))
    newBoxActor = rigid.addBox(newbox, rigidBody)
    hingeJoint = rigid.createHingeJoint(oldBoxActor, newBoxActor)
    hingeJoint.setAnchorPoint((oldbox.center + newbox.center) / 2)
    hingeJoint.setAxis(dyno.Vector3f([0,0,1]))
    hingeJoint.setRange(-3.14159265358979323846, 3.14159265358979323846)
    oldbox = newbox
    if i == 19:
        pointJoint = rigid.createPointJoint(newBoxActor)
        pointJoint.setAnchorPoint(newbox.center)

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
wireRender.setColor(dyno.Color(0, 0, 0))
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
