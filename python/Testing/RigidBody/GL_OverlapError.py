import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)
rA = rB = rC = rD = dyno.RigidBodyInfo()
box1 = box2 = box3 =  dyno.BoxInfo()

rA.position = dyno.Vector3f([0,0.5,0])
box1.half_length = dyno.Vector3f([1,0.05,1])

rB.position = dyno.Vector3f([0,0.2,0.8])
box2.half_length = dyno.Vector3f([0.7,0.1,0.1])

rC.position = dyno.Vector3f([0,0.2,-0.8])
box3.half_length = dyno.Vector3f([0.7,0.1,0.1])

bodyActor = rigid.add_box(box1, rA)
frontActor = rigid.add_box(box2, rB)
rearActor = rigid.add_box(box3, rC)

sphere = dyno.SphereInfo()
sphere.radius = 0.1

rA.position = dyno.Vector3f([0.9,0.1,0.8])
rB.position = dyno.Vector3f([-0.9,0.1,0.8])
rC.position = dyno.Vector3f([0.9,0.1,-0.8])
rD.position = dyno.Vector3f([-0.9,0.1,-0.8])

frontLeftTire = rigid.add_sphere(sphere, rA)
frontRightTire = rigid.add_sphere(sphere, rB)
rearLeftTire = rigid.add_sphere(sphere, rC)
rearRightTire = rigid.add_sphere(sphere, rD)

joint1 = rigid.create_hinge_joint(frontLeftTire, frontActor)
joint1.set_anchor_point(frontLeftTire.center)
joint1.set_axis(dyno.Vector3f([1,0,0]))
joint1.set_moter(30)

joint2 = rigid.create_hinge_joint(frontRightTire, frontActor)
joint2.set_anchor_point(frontRightTire.center)
joint2.set_axis(dyno.Vector3f([1,0,0]))
joint2.set_moter(30)

joint3 = rigid.create_hinge_joint(rearLeftTire, rearActor)
joint3.set_anchor_point(rearLeftTire.center)
joint3.set_axis(dyno.Vector3f([1,0,0]))
joint3.set_moter(30)

joint4 = rigid.create_hinge_joint(rearRightTire, rearActor)
joint4.set_anchor_point(rearRightTire.center)
joint4.set_axis(dyno.Vector3f([1,0,0]))
joint4.set_moter(30)

joint5 = rigid.create_fixed_joint(rearActor, bodyActor)
joint5.set_anchor_point(rearActor.center)

joint6 = rigid.create_hinge_joint(frontActor, bodyActor)
joint6.set_anchor_point(frontActor.center)
joint6.set_axis(dyno.Vector3f([0,1,0]))
joint6.set_range(M_PI / 12, M_PI / 12)

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
wireRender.set_color(dyno.Color(0, 0, 1))
contactMapper.out_edge_set().connect(wireRender.in_edge_set())
rigid.graphics_pipeline().push_module(wireRender)

# contactPointMapper = dyno.ContactsToPointSet3f()
# elementQuery.out_contacts().connect(contactPointMapper.in_contacts())
# rigid.graphics_pipeline().push_module(contactPointMapper)
#
# pointRender = dyno.GLPointVisualModule()
# pointRender.set_color(dyno.Color(1, 0, 0))
# pointRender.var_point_size().set_value(0.003)
# contactPointMapper.out_point_set().connect(pointRender.in_point_set())
# rigid.graphics_pipeline().push_module(pointRender)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
