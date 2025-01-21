import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)
dim = 5
h = 0.1

rigidBody = dyno.RigidBodyInfo()
box = dyno.BoxInfo()
box.half_length = dyno.Vector3f([h,h,h])

test = [dyno.BodyType.Dynamic, dyno.BodyType.Kinematic, dyno.BodyType.Static]

cnt =1

for te in test:
    rigidBody.angular_velocity = dyno.Vector3f([0,0,0.5])
    rigidBody.motion_type = te
    rigidBody.position = dyno.Vector3f([cnt*0.5,0.5,0])
    cnt = cnt + 1
    boxAt = rigid.add_box(box, rigidBody)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0.36078, 0.67451, 0.93333))
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
