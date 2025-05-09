import PyPeridyno as dyno


def createTwoBoxes(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.position = dyno.Vector3f([-0.3, 0.1, 0.5])
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])

    box = dyno.BoxInfo()
    box.half_length = dyno.Vector3f([0.1, 0.1, 0.1])
    rigid.add_box(box, rigidBody)

    rigidBody.position = dyno.Vector3f([-0.3, 0.3, 0.5])
    rigidBody.motion_type = dyno.BodyType.Dynamic
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])

    box.half_length = dyno.Vector3f([0.1, 0.1, 0.1])
    rigid.add_box(box, rigidBody)


def createTwoTets(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])

    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([0.45, 0.3, 0.45]),
        dyno.Vector3f([0.45, 0.55, 0.45]),
        dyno.Vector3f([0.7, 0.3, 0.45]),
        dyno.Vector3f([0.45, 0.3, 0.7]),
    ]
    rigid.add_tet(tet0, rigidBody)

    tet1 = dyno.TetInfo()
    tet1.v = [
        dyno.Vector3f([0.45, 0, 0.45]),
        dyno.Vector3f([0.45, 0.25, 0.45]),
        dyno.Vector3f([0.7, 0, 0.45]),
        dyno.Vector3f([0.45, 0, 0.7]),
    ]
    rigid.add_tet(tet1, rigidBody)


def createTetBox(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.position = dyno.Vector3f([1.3, 0.1, 0.5])
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    box = dyno.BoxInfo()
    box.half_length = dyno.Vector3f([0.1, 0.1, 0.1])
    rigid.add_box(box, rigidBody)

    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([1.25, 0.25, 0.45]),
        dyno.Vector3f([1.25, 0.5, 0.45]),
        dyno.Vector3f([1.5, 0.25, 0.45]),
        dyno.Vector3f([1.25, 0.25, 0.7]),
    ]
    rigid.add_tet(tet0, rigidBody)


def createTwoCapsules(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.position = dyno.Vector3f([-1.25, 0.1, -0.5])
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])

    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.half_length = 0.1
    capsule.radius = 0.1
    rigid.add_capsule(capsule, rigidBody)

    rigidBody.position = dyno.Vector3f([-1.3, 0.3, -0.5])
    capsule.half_length = 0.1
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.radius = 0.1
    rigid.add_capsule(capsule, rigidBody)


def createCapsuleBox(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    box = dyno.BoxInfo()
    box.half_length = dyno.Vector3f([0.1, 0.1, 0.1])

    rigidBody.position = dyno.Vector3f([-1.3, 0.1, 0.5])

    rigid.add_box(box, rigidBody)

    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.half_length = 0.1
    capsule.radius = 0.1

    rigidBody.position = dyno.Vector3f([-1.3, 0.3, 0.5])

    rigid.add_capsule(capsule, rigidBody)

def createBoxCapsule(rigid):
    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.half_length = 0.1
    capsule.radius = 0.1

    rigidBody = dyno.RigidBodyInfo()
    rigidBody.position = dyno.Vector3f([-0.3, 0.1, -0.35])
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])

    rigid.add_capsule(capsule, rigidBody)

    box = dyno.BoxInfo()
    box.half_length = dyno.Vector3f([0.1, 0.1, 0.1])

    rigidBody.position = dyno.Vector3f([-0.3, 0.3, -0.5])

    rigid.add_box(box, rigidBody)

def createCapsuleTet(rigid):
    rigidBody = dyno.RigidBodyInfo()

    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([0.45, 0, -0.45]),
        dyno.Vector3f([0.45, 0.25, -0.45]),
        dyno.Vector3f([0.7, 0, -0.45]),
        dyno.Vector3f([0.45, 0, -0.2]),
    ]
    rigid.add_tet(tet0, rigidBody)

    capsule = dyno.CapsuleInfo()

    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.half_length = 0.1
    capsule.radius = 0.1

    rigidBody.position = dyno.Vector3f([0.45, 0.4, -0.35])

    rigid.add_capsule(capsule, rigidBody)

def createTetCapsule(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])

    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([1.25, 0.3, -0.45]),
        dyno.Vector3f([1.25, 0.55, -0.45]),
        dyno.Vector3f([1.5, 0.3, -0.45]),
        dyno.Vector3f([1.25, 0.3, -0.2]),
    ]
    rigid.add_tet(tet0, rigidBody)

    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.half_length = 0.1
    capsule.radius = 0.1

    rigidBody.position = dyno.Vector3f([1.25, 0.1, -0.35])

    rigid.add_capsule(capsule, rigidBody)

def createTwoSpheres(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    sphere = dyno.SphereInfo()
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([-1.3, 0.1, 1.5])
    rigid.add_sphere(sphere, rigidBody)

    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([-1.3, 0.3, 1.59])

    rigid.add_sphere(sphere, rigidBody)


def createSphereBox(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    box = dyno.BoxInfo()
    box.half_length = dyno.Vector3f([0.1, 0.1, 0.1])

    rigidBody.position = dyno.Vector3f([-0.3, 0.3, 1.5])

    rigid.add_box(box, rigidBody)

    sphere = dyno.SphereInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([-0.3, 0.1, 1.59])

    rigid.add_sphere(sphere, rigidBody)


def createSphereTet(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    tet0 = dyno.TetInfo()
    tet0.v = [
        dyno.Vector3f([0.45, 0.3, 1.59]),
        dyno.Vector3f([0.45, 0.55, 1.59]),
        dyno.Vector3f([0.7, 0.3, 1.59]),
        dyno.Vector3f([0.45, 0.3, 1.89]),
    ]
    rigid.add_tet(tet0, rigidBody)

    sphere = dyno.SphereInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([0.7, 0.1, 1.59])

    rigid.add_sphere(sphere, rigidBody)


def createSphereCapsule(rigid):
    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    capsule = dyno.CapsuleInfo()
    capsule.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([1, 0, 0]))
    capsule.half_length = 0.1
    capsule.radius = 0.1

    rigidBody.position = dyno.Vector3f([1.3, 0.3, 1.6])

    rigid.add_capsule(capsule, rigidBody)

    sphere = dyno.SphereInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])
    sphere.radius = 0.1

    rigidBody.position = dyno.Vector3f([1.3, 0.1, 1.59])

    rigid.add_sphere(sphere, rigidBody)


scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)
rigid.set_dt(0.005)
mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_alpha(0.8)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
rigid.graphics_pipeline().push_module(sRender)

elementQuery = dyno.NeighborElementQuery3f()
rigid.state_topology().connect(elementQuery.in_discrete_elements())
rigid.state_collision_mask().connect(elementQuery.in_collision_mask())
rigid.graphics_pipeline().push_module(elementQuery)

contactMapper = dyno.ContactsToEdgeSet3f()
elementQuery.out_contacts().connect(contactMapper.in_contacts())
contactMapper.var_scale().set_value(0.02)
rigid.graphics_pipeline().push_module(contactMapper)

wireRender = dyno.GLWireframeVisualModule()
wireRender.set_color(dyno.Color(0, 0, 1))
contactMapper.out_edge_set().connect(wireRender.in_edge_set())
rigid.graphics_pipeline().push_module(wireRender)

contactPointMapper = dyno.ContactsToPointSet3f()
elementQuery.out_contacts().connect(contactPointMapper.in_contacts())
rigid.graphics_pipeline().push_module(contactPointMapper)

pointRender = dyno.GLPointVisualModule()
pointRender.set_color(dyno.Color(1, 0, 0))
pointRender.var_point_size().set_value(0.01)
contactPointMapper.out_point_set().connect(pointRender.in_point_set())
rigid.graphics_pipeline().push_module(pointRender)

cdBV = dyno.CollistionDetectionBoundingBox3f()
rigid.state_topology().connect(cdBV.in_discrete_elements())
rigid.graphics_pipeline().push_module(cdBV)

boundaryContactsMapper = dyno.ContactsToPointSet3f()
cdBV.out_contacts().connect(boundaryContactsMapper.in_contacts())
rigid.graphics_pipeline().push_module(boundaryContactsMapper)

boundaryContactsRender = dyno.GLPointVisualModule()
boundaryContactsRender.set_color(dyno.Color(0, 1, 0))
boundaryContactsRender.var_point_size().set_value(0.01)
boundaryContactsMapper.out_point_set().connect(boundaryContactsRender.in_point_set())
rigid.graphics_pipeline().push_module(boundaryContactsRender)

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
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
