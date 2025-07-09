import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)
actor = rigid.createRigidBody(dyno.Vector3f([0,0.3,0]), dyno.Quat1f(0.5, dyno.Vector3f([1,0,1])))
box = dyno.BoxInfo()
box.center = dyno.Vector3f([0.05,0.05,0.05])
rigid.bindBox(actor, box, 100)

sphere = dyno.SphereInfo()
sphere.center = dyno.Vector3f([0,0,0.1])
sphere.radius = 0.1
rigid.bindSphere(actor, sphere, 100)

capsule = dyno.CapsuleInfo()
capsule.center = dyno.Vector3f([-0.15, 0,0])
capsule.radius = 0.1
capsule.halfLength = 0.1
rigid.bindCapsule(actor, capsule, 100)

actor2 = rigid.createRigidBody(dyno.Vector3f([-0.1,0.6,0]), dyno.Quat1f())
sphere2 = dyno.SphereInfo()
sphere2.center = dyno.Vector3f([0,0,0])
sphere2.radius = 0.1
rigid.bindSphere(actor2, sphere2, 100)



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
