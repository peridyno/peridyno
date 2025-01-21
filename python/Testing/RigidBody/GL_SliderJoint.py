import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

rigidBody = dyno.RigidBodyInfo()
rigidBody.friction = 0.01
box1 = box2 = dyno.BoxInfo()

box1.half_length = dyno.Vector3f([0.09, 0.1, 0.1])

rigidBody.position = dyno.Vector3f([0, 0.5,0])

boxActor1 = rigid.add_box(box1, rigidBody, 1000)

box2.half_length = dyno.Vector3f([0.1, 0.4,0.2])

rigidBody.position = dyno.Vector3f([0.2, 0.5, 0])
rigidBody.angular_velocity = dyno.Vector3f([1,0,0])
rigidBody.motion_type = dyno.BodyType.Kinematic

boxActor2 = rigid.add_box(box2, rigidBody)

sliderJoint = rigid.create_slider_joint(boxActor1, boxActor2)

sliderJoint.set_anchor_point((boxActor1.center + boxActor2.center) / 2)

sliderJoint.set_axis(dyno.Vector3f([0,1,0]))
sliderJoint.set_range(-0.2, 0.2)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(1,1,0))
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
