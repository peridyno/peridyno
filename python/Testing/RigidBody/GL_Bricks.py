import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)
rigidBody = dyno.RigidBodyInfo()
rigidBody.linearVelocity = dyno.Vector3f([1, 0, 0])

box = dyno.BoxInfo()
box.halfLength = dyno.Vector3f([0.5 * 0.065, 0.5 * 0.065, 0.5 * 0.1])
for i in range(8, 1, -1):
    for j in range(i + 1):
        rigidBody.position = dyno.Vector3f([0.5, 0.5, 0.5]) * dyno.Vector3f(
            [0.5, 1.1 - 0.13 * i, 0.12 + 0.2 * j + 0.1 * (8 - i)])
        boxAt = rigid.addBox(box, rigidBody)

for i in range(8, 1, -1):
    for j in range(i+1):
        rigidBody.position = dyno.Vector3f([0.5, 0.5, 0.5]) * dyno.Vector3f(
            [2.5, 1.1 - 0.13 * i, 0.12 + 0.2 * j + 0.1 * (8 - i)])
        rigidBody.friction = 0.1
        boxAt = rigid.addBox(box, rigidBody)

sphere = dyno.SphereInfo()
sphere.radius = 0.025

rigidSphere = dyno.RigidBodyInfo()
rigidSphere.position = dyno.Vector3f([0.5, 0.75, 0.5])
sphereAt1 = rigid.addSphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.95, 0.5])
sphereAt2 = rigid.addSphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.65, 0.5])
sphere.radius = 0.05
sphereAt3 = rigid.addSphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0, 0, 0])
tet = dyno.TetInfo()
tet.v = [
    dyno.Vector3f([0.5, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.2, 0.5]),
    dyno.Vector3f([0.6, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.1, 0.6]),
]
TetAt = rigid.addTet(tet, rigidSphere)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(1, 1, 0))
sRender.setAlpha(0.5)
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
