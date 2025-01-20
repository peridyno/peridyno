import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

dim = 5
h = 0.1

rigidBody = dyno.RigidBodyInfo()
rigidBody.linear_velocity = dyno.Vector3f([0.0, 0, 0])
box = dyno.BoxInfo()
box.half_length = dyno.Vector3f([h, h, h])
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            rigidBody.position = dyno.Vector3f([2 * i * h - h * dim, h + 2.01 * j * h, 2 * k * h - h * dim])
            boxAt = rigid.add_box(box, rigidBody)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0.36078, 0.67451, 0.93333))
sRender.set_alpha(1)
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
mapper.out_triangle_set().connect(wireRender.in_edge_set())
rigid.graphics_pipeline().push_module(wireRender)



app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
