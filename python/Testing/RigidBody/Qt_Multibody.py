import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

def createCompound(scene):
    rigid = dyno.RigidBodySystem3f()
    scene.addNode(rigid)
    actor = rigid.createRigidBody(dyno.Vector3f([0, 1.3, 0]), dyno.Quat1f(0.5, dyno.Vector3f([1, 0, 1])))
    box = dyno.BoxInfo()
    box.center = dyno.Vector3f([0.15, 0, 0])
    box.halfLength = dyno.Vector3f([0.05, 0.05, 0.05])
    rigid.bindBox(actor, box, 100)

    sphere = dyno.SphereInfo()
    sphere.center = dyno.Vector3f([0, 0, 0.1])
    sphere.radius = 0.1
    rigid.bindSphere(actor, sphere, 100)

    capsule = dyno.CapsuleInfo()
    capsule.center = dyno.Vector3f([-0.15, 0, 0])
    capsule.radius = 0.1
    capsule.halfLength = 0.1
    rigid.bindCapsule(actor, capsule, 100)

    actor2 = rigid.createRigidBody(dyno.Vector3f([-0.1, 1.6, 0]), dyno.Quat1f())
    sphere2 = dyno.SphereInfo()
    sphere2.center = dyno.Vector3f([0, 0, 0])
    sphere2.radius = 0.1
    rigid.bindSphere(actor2, sphere2, 100)

    mapper = dyno.DiscreteElementsToTriangleSet3f()
    rigid.stateTopology().connect(mapper.inDiscreteElements())
    rigid.graphicsPipeline().pushModule(mapper)

    sRender = dyno.GLSurfaceVisualModule()
    sRender.setColor(dyno.Color(1, 1, 0))
    sRender.setAlpha(0.5)
    mapper.outTriangleSet().connect(sRender.inTriangleSet())
    rigid.graphicsPipeline().pushModule(sRender)

    return rigid

def createBoxes(scene):
    rigid = dyno.RigidBodySystem3f()
    scn.addNode(rigid)

    dim = 2
    h = 0.1

    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linearVelocity = dyno.Vector3f([0.0, 0, 0])
    box = dyno.BoxInfo()
    box.halfLength = dyno.Vector3f([h, h, h])
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                rigidBody.position = dyno.Vector3f([2 * i * h - h * dim, h + 2.01 * j * h, 2 * k * h - h * dim])
                rigidBody.motionType = dyno.BodyType.Static
                boxAt = rigid.addBox(box, rigidBody)

    mapper = dyno.DiscreteElementsToTriangleSet3f()
    rigid.stateTopology().connect(mapper.inDiscreteElements())
    rigid.graphicsPipeline().pushModule(mapper)

    sRender = dyno.GLSurfaceVisualModule()
    sRender.setColor(dyno.Color(0.36078, 0.67451, 0.93333))
    sRender.setAlpha(1)
    mapper.outTriangleSet().connect(sRender.inTriangleSet())
    rigid.graphicsPipeline().pushModule(sRender)

    wireRender = dyno.GLWireframeVisualModule()
    wireRender.setColor(dyno.Color(0, 0, 0))
    mapper.outTriangleSet().connect(wireRender.inEdgeSet())
    rigid.graphicsPipeline().pushModule(wireRender)

    return rigid


compound = createCompound(scn)
boxes = createBoxes(scn)

plane = dyno.PlaneModel3f()
scn.addNode(plane)
plane.varLengthX().setValue(5)
plane.varLengthZ().setValue(5)

convoy = dyno.MultibodySystem3f()
scn.addNode(convoy)
boxes.connect(convoy.importVehicles())
compound.connect(convoy.importVehicles())

plane.stateTriangleSet().connect(convoy.inTriangleSet())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()
