import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

rA = dyno.RigidBodyInfo(dyno.Vector3f([0, 0.8, 0]), dyno.Quat1f(0,0,0,1))
rB = dyno.RigidBodyInfo(dyno.Vector3f([0, 0.3, 0]), dyno.Quat1f(0,0,0,1))
rC = dyno.RigidBodyInfo(dyno.Vector3f([-0.26, 0.55, 0]), dyno.Quat1f(0,0,0,1))
rD = dyno.RigidBodyInfo(dyno.Vector3f([0.26, 0.55, 0]), dyno.Quat1f(0,0,0,1))

box1 = box2 = box3 = box4 =  dyno.BoxInfo()
box1.half_length = dyno.Vector3f([0.03,0.2,0.03])
box1.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([0,0,1]))

box2.half_length = dyno.Vector3f([0.03,0.2,0.03])
box2.rot = dyno.Quat1f(3.14159265358979323846 / 2, dyno.Vector3f([0,0,1]))

box3.half_length = dyno.Vector3f([0.03,0.2,0.03])

box4.half_length = dyno.Vector3f([0.03,0.2,0.03])

boxActor1 = rigid.add_box(box1, rA)
boxActor2 = rigid.add_box(box2, rB)
boxActor3 = rigid.add_box(box3, rC)
boxActor4 = rigid.add_box(box4, rD)

hingeJoint1 = rigid.create_hinge_joint(boxActor1, boxActor3)
hingeJoint1.set_anchor_point(dyno.Vector3f([-0.26, 0.8, 0]))
hingeJoint1.set_axis(dyno.Vector3f([0,0,1]))
hingeJoint1.set_range(-3.14159265358979323846 * 2 / 3, 3.14159265358979323846 * 2 / 3)
hingeJoint2 = rigid.create_hinge_joint(boxActor1, boxActor4)
hingeJoint2.set_anchor_point(dyno.Vector3f([0.26, 0.8, 0]))
hingeJoint2.set_axis(dyno.Vector3f([0,0,1]))
hingeJoint2.set_range(-3.14159265358979323846 * 2 / 3, 3.14159265358979323846 * 2 / 3)
hingeJoint3 = rigid.create_hinge_joint(boxActor2, boxActor3)
hingeJoint3.set_anchor_point(dyno.Vector3f([-0.26, 0.3, 0]))
hingeJoint3.set_axis(dyno.Vector3f([0,0,1]))
hingeJoint3.set_range(-3.14159265358979323846 * 2 / 3, 3.14159265358979323846 * 2 / 3)
hingeJoint4 = rigid.create_hinge_joint(boxActor2, boxActor4)
hingeJoint4.set_anchor_point(dyno.Vector3f([0.26, 0.3, 0]))
hingeJoint4.set_axis(dyno.Vector3f([0,0,1]))
hingeJoint4.set_range(-3.14159265358979323846 * 2 / 3, 3.14159265358979323846 * 2 / 3)

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
