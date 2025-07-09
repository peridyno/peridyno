import PyPeridyno as dyno


def createTwoBoxes(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.position = dyno.Vector3f([-0.3, 0.1, 0.5])
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])

    box = dyno.BoxInfo()
    box.halfLength = dyno.Vector3f([0.1, 0.1, 0.1])
    rigid.addBox(box, rigidBody)

    rigidBody.position = dyno.Vector3f([-0.3, 0.3, 0.5])
    rigidBody.motionType = dyno.BodyType.Dynamic
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])

    box.halfLength = dyno.Vector3f([0.1, 0.1, 0.1])
    rigid.addBox(box, rigidBody)


def createTwoTets(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])

    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([0.45, 0.3, 0.45]),
        dyno.Vector3f([0.45, 0.55, 0.45]),
        dyno.Vector3f([0.7, 0.3, 0.45]),
        dyno.Vector3f([0.45, 0.3, 0.7]),
    ]
    rigid.addTet(tet0, rigidBody)

    tet1 = dyno.TetInfo()
    tet1.v = [
        dyno.Vector3f([0.45, 0, 0.45]),
        dyno.Vector3f([0.45, 0.25, 0.45]),
        dyno.Vector3f([0.7, 0, 0.45]),
        dyno.Vector3f([0.45, 0, 0.7]),
    ]
    rigid.addTet(tet1, rigidBody)


def createTetBox(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.position = dyno.Vector3f([1.3, 0.1, 0.5])
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    box = dyno.BoxInfo()
    box.halfLength = dyno.Vector3f([0.1, 0.1, 0.1])
    rigid.addBox(box, rigidBody)

    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([1.25, 0.25, 0.45]),
        dyno.Vector3f([1.25, 0.5, 0.45]),
        dyno.Vector3f([1.5, 0.25, 0.45]),
        dyno.Vector3f([1.25, 0.25, 0.7]),
    ]
    rigid.addTet(tet0, rigidBody)


def createTwoCapsules(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.position = dyno.Vector3f([-1.25, 0.1, -0.5])
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])

    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.halfLength = 0.1
    capsule.radius = 0.1
    rigid.addCapsule(capsule, rigidBody)

    rigidBody.position = dyno.Vector3f([-1.3, 0.3, -0.5])
    capsule.halfLength = 0.1
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.radius = 0.1
    rigid.addCapsule(capsule, rigidBody)


def createCapsuleBox(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    box = dyno.BoxInfo()
    box.halfLength = dyno.Vector3f([0.1, 0.1, 0.1])

    rigidBody.position = dyno.Vector3f([-1.3, 0.1, 0.5])

    rigid.addBox(box, rigidBody)

    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.halfLength = 0.1
    capsule.radius = 0.1

    rigidBody.position = dyno.Vector3f([-1.3, 0.3, 0.5])

    rigid.addCapsule(capsule, rigidBody)

def createBoxCapsule(rigid):
    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.halfLength = 0.1
    capsule.radius = 0.1

    rigidBody = dyno.RigidBodyInfo()
    rigidBody.position = dyno.Vector3f([-0.3, 0.1, -0.35])
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])

    rigid.addCapsule(capsule, rigidBody)

    box = dyno.BoxInfo()
    box.halfLength = dyno.Vector3f([0.1, 0.1, 0.1])

    rigidBody.position = dyno.Vector3f([-0.3, 0.3, -0.5])

    rigid.addBox(box, rigidBody)

def createCapsuleTet(rigid):
    rigidBody = dyno.RigidBodyInfo()

    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([0.45, 0, -0.45]),
        dyno.Vector3f([0.45, 0.25, -0.45]),
        dyno.Vector3f([0.7, 0, -0.45]),
        dyno.Vector3f([0.45, 0, -0.2]),
    ]
    rigid.addTet(tet0, rigidBody)

    capsule = dyno.CapsuleInfo()

    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.halfLength = 0.1
    capsule.radius = 0.1

    rigidBody.position = dyno.Vector3f([0.45, 0.4, -0.35])

    rigid.addCapsule(capsule, rigidBody)

def createTetCapsule(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])

    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([1.25, 0.3, -0.45]),
        dyno.Vector3f([1.25, 0.55, -0.45]),
        dyno.Vector3f([1.5, 0.3, -0.45]),
        dyno.Vector3f([1.25, 0.3, -0.2]),
    ]
    rigid.addTet(tet0, rigidBody)

    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.halfLength = 0.1
    capsule.radius = 0.1

    rigidBody.position = dyno.Vector3f([1.25, 0.1, -0.35])

    rigid.addCapsule(capsule, rigidBody)

def createTwoSpheres(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    sphere = dyno.SphereInfo()
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([-1.3, 0.1, 1.5])
    rigid.addSphere(sphere, rigidBody)

    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([-1.3, 0.3, 1.59])

    rigid.addSphere(sphere, rigidBody)


def createSphereBox(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    box = dyno.BoxInfo()
    box.halfLength = dyno.Vector3f([0.1, 0.1, 0.1])

    rigidBody.position = dyno.Vector3f([-0.3, 0.3, 1.5])

    rigid.addBox(box, rigidBody)

    sphere = dyno.SphereInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([-0.3, 0.1, 1.59])

    rigid.addSphere(sphere, rigidBody)


def createSphereTet(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([0.45, 0.3, 1.59]),
        dyno.Vector3f([0.45, 0.55, 1.59]),
        dyno.Vector3f([0.7, 0.3, 1.59]),
        dyno.Vector3f([0.45, 0.3, 1.89]),
    ]
    rigid.addTet(tet0, rigidBody)

    sphere = dyno.SphereInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([0.7, 0.1, 1.59])

    rigid.addSphere(sphere, rigidBody)


def createSphereCapsule(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.halfLength = 0.1
    capsule.radius = 0.1

    rigidBody.position = dyno.Vector3f([1.3, 0.3, 1.6])

    rigid.addCapsule(capsule, rigidBody)

    sphere = dyno.SphereInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0, 0, 0])
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([1.3, 0.1, 1.59])

    rigid.addSphere(sphere, rigidBody)


scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)
rigid.setDt(0.005)
mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setAlpha(0.8)
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
pointRender.varPointSize().setValue(0.01)
contactPointMapper.outPointSet().connect(pointRender.inPointSet())
rigid.graphicsPipeline().pushModule(pointRender)

cdBV = dyno.CollistionDetectionBoundingBox3f()
rigid.stateTopology().connect(cdBV.inDiscreteElements())
rigid.graphicsPipeline().pushModule(cdBV)

boundaryContactsMapper = dyno.ContactsToPointSet3f()
cdBV.outContacts().connect(boundaryContactsMapper.inContacts())
rigid.graphicsPipeline().pushModule(boundaryContactsMapper)

boundaryContactsRender = dyno.GLPointVisualModule()
boundaryContactsRender.setColor(dyno.Color(0, 1, 0))
boundaryContactsRender.varPointSize().setValue(0.01)
boundaryContactsMapper.outPointSet().connect(boundaryContactsRender.inPointSet())
rigid.graphicsPipeline().pushModule(boundaryContactsRender)

createTwoBoxes(rigid)
createTwoTets(rigid)
createTetBox(rigid)
createCapsuleBox(rigid)
createTwoCapsules(rigid)
createCapsuleTet(rigid)
createBoxCapsule(rigid)
createTetCapsule(rigid)

createTwoSpheres(rigid)
createSphereBox(rigid)
createSphereTet(rigid)
createSphereCapsule(rigid)


app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()
