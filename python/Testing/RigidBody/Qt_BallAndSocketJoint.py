import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

box =  dyno.BoxInfo()
rA = dyno.RigidBodyInfo()
rA.bodyId = 1
rA.linear_velocity = dyno.Vector3f([1,0,0])
box.center = dyno.Vector3f([0,0,0])
box.half_length = dyno.Vector3f([0.05,0.05,0.05])
oldBoxActor = rigid.add_box(box, rA)

for i in range(100):
    rB = dyno.RigidBodyInfo()
    rB.position = rA.position + dyno.Vector3f([0, 0.12, 0])
    rB.linear_velocity = dyno.Vector3f([0,0,0])

    newBoxActor = rigid.add_box(box, rB)

    ballAndSockerJoint = rigid.create_ball_and_socket_joint(oldBoxActor, newBoxActor)
    ballAndSockerJoint.set_anchor_point((rA.position + rB.position) / 2)

    rA = rB
    oldBoxActor = newBoxActor

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0.3,0.5,0.9))
sRender.set_alpha(0.8)
sRender.set_roughness(0.7)
sRender.set_metallic(3)
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
