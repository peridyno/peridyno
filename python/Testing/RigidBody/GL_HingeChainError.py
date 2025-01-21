import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

rigidBody = dyno.RigidBodyInfo()
scale = 2.5
sphere = dyno.SphereInfo()
sphere.center = dyno.Vector3f([scale, scale, scale]) * dyno.Vector3f([-4.6,20,0.5])
sphere.radius = scale * 2.5

newbox = oldbox = dyno.BoxInfo()
oldbox.center = dyno.Vector3f([scale, scale, scale]) * dyno.Vector3f([-2.0,20,0.5])
oldbox.half_length = dyno.Vector3f([scale, scale, scale]) * dyno.Vector3f([0.05,0.09,0.02])
oldbox.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([0,0,0]))
oldBoxActor = rigid.add_box(oldbox, rigidBody)
rigidBody.linear_velocity = dyno.Vector3f([0,0,0])

for i in range(20):
    newbox.center = oldbox.center + dyno.Vector3f([scale, scale, scale]) * dyno.Vector3f([0.2,0,0])
    newbox.half_length = oldbox.half_length
    newbox.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([0,0,1]))
    newBoxActor = rigid.add_box(newbox, rigidBody)
    hingeJoint = rigid.create_hinge_joint(oldBoxActor, newBoxActor)
    hingeJoint.set_anchor_point((oldbox.center + newbox.center) / 2)
    hingeJoint.set_axis(dyno.Vector3f([0,0,1]))
    hingeJoint.set_range(-3.14159265358979323846, 3.14159265358979323846)
    oldbox = newbox
    if i == 19:
        pointJoint = rigid.create_point_joint(newBoxActor)
        pointJoint.set_anchor_point(newbox.center)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0.204, 0.424, 0.612))
sRender.set_alpha(1.0)
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
wireRender.set_color(dyno.Color(0, 0, 0))
contactMapper.out_edge_set().connect(wireRender.in_edge_set())
rigid.graphics_pipeline().push_module(wireRender)

contactPointMapper = dyno.ContactsToPointSet3f()
elementQuery.out_contacts().connect(contactPointMapper.in_contacts())
rigid.graphics_pipeline().push_module(contactPointMapper)

pointRender = dyno.GLPointVisualModule()
pointRender.set_color(dyno.Color(1, 0, 0))
pointRender.var_point_size().set_value(0.003)
contactPointMapper.out_point_set().connect(pointRender.in_point_set())
rigid.graphics_pipeline().push_module(pointRender)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
