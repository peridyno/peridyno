import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

rA = dyno.RigidBodyInfo()
box =  dyno.BoxInfo()
box.center = dyno.Vector3f([0,0,0])
box.half_length = dyno.Vector3f([2,2,2]) * dyno.Vector3f([0.02, 0.08,0.02])
rA.position = dyno.Vector3f([2,2,2]) * dyno.Vector3f([-1,0.2,0.5])
rA.linear_velocity = dyno.Vector3f([1,0,0])
oldBoxActor = rigid.add_box(box, rA)
rA.linear_velocity = dyno.Vector3f([0,0,0])

for i in range(10):
    rB = dyno.RigidBodyInfo()
    rB.position = rA.position + dyno.Vector3f([2,2,2]) *  dyno.Vector3f([0, 0.2, 0])

    newBoxActor = rigid.add_box(box, rB)
    hingeJoint = rigid.create_hinge_joint(oldBoxActor, newBoxActor)
    hingeJoint.set_anchor_point((rA.position + rB.position) / 2)
    hingeJoint.set_axis(dyno.Vector3f([0,0,1]))
    hingeJoint.set_range(-3.14159265358979323846 / 2, 3.14159265358979323846 / 2)

    rA = rB
    oldBoxActor = newBoxActor

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(1, 1, 0))
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
wireRender.set_color(dyno.Color(0, 0, 1))
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
