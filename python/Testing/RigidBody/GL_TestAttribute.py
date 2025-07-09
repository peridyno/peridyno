import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)
dim = 5
h = 0.1

rigidBody = dyno.RigidBodyInfo()
box = dyno.BoxInfo()
box.halfLength = dyno.Vector3f([h,h,h])

test = [dyno.BodyType.Dynamic, dyno.BodyType.Kinematic, dyno.BodyType.Static]

cnt =1

for te in test:
    rigidBody.angularVelocity = dyno.Vector3f([0,0,0.5])
    rigidBody.motionType = te
    rigidBody.position = dyno.Vector3f([cnt*0.5,0.5,0])
    cnt = cnt + 1
    boxAt = rigid.addBox(box, rigidBody)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(0.36078, 0.67451, 0.93333))
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
