import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)
actor = rigid.create_rigid_body(dyno.Vector3f([0,0.3,0]), dyno.Quat1f(0.5, dyno.Vector3f([1,0,1])))
box = dyno.BoxInfo()
box.center = dyno.Vector3f([0.05,0.05,0.05])
rigid.bind_box(actor, box, 100)

sphere = dyno.SphereInfo()
sphere.center = dyno.Vector3f([0,0,0.1])
sphere.radius = 0.1
rigid.bind_sphere(actor, sphere, 100)

capsule = dyno.CapsuleInfo()
capsule.center = dyno.Vector3f([-0.15, 0,0])
capsule.radius = 0.1
capsule.half_length = 0.1
rigid.bind_capsule(actor, capsule, 100)

actor2 = rigid.create_rigid_body(dyno.Vector3f([-0.1,0.6,0]), dyno.Quat1f())
sphere2 = dyno.SphereInfo()
sphere2.center = dyno.Vector3f([0,0,0])
sphere2.radius = 0.1
rigid.bind_sphere(actor2, sphere2, 100)



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
