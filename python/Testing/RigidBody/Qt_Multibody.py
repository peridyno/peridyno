import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

def createCompound(scene):
    rigid = dyno.RigidBodySystem3f()
    scene.add_node(rigid)
    actor = rigid.create_rigid_body(dyno.Vector3f([0, 1.3, 0]), dyno.Quat1f(0.5, dyno.Vector3f([1, 0, 1])))
    box = dyno.BoxInfo()
    box.center = dyno.Vector3f([0.15, 0, 0])
    box.half_length = dyno.Vector3f([0.05, 0.05, 0.05])
    rigid.bind_box(actor, box, 100)

    sphere = dyno.SphereInfo()
    sphere.center = dyno.Vector3f([0, 0, 0.1])
    sphere.radius = 0.1
    rigid.bind_sphere(actor, sphere, 100)

    capsule = dyno.CapsuleInfo()
    capsule.center = dyno.Vector3f([-0.15, 0, 0])
    capsule.radius = 0.1
    capsule.half_length = 0.1
    rigid.bind_capsule(actor, capsule, 100)

    actor2 = rigid.create_rigid_body(dyno.Vector3f([-0.1, 1.6, 0]), dyno.Quat1f())
    sphere2 = dyno.SphereInfo()
    sphere2.center = dyno.Vector3f([0, 0, 0])
    sphere2.radius = 0.1
    rigid.bind_sphere(actor2, sphere2, 100)

    mapper = dyno.DiscreteElementsToTriangleSet3f()
    rigid.state_topology().connect(mapper.in_discrete_elements())
    rigid.graphics_pipeline().push_module(mapper)

    sRender = dyno.GLSurfaceVisualModule()
    sRender.set_color(dyno.Color(1, 1, 0))
    sRender.set_alpha(0.5)
    mapper.out_triangle_set().connect(sRender.in_triangle_set())
    rigid.graphics_pipeline().push_module(sRender)

    return rigid

def createBoxes(scene):
    rigid = dyno.RigidBodySystem3f()
    scn.add_node(rigid)

    dim = 2
    h = 0.1

    rigidBody = dyno.RigidBodyInfo()
    rigidBody.linear_velocity = dyno.Vector3f([0.0, 0, 0])
    box = dyno.BoxInfo()
    box.half_length = dyno.Vector3f([h, h, h])
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                rigidBody.position = dyno.Vector3f([2 * i * h - h * dim, h + 2.01 * j * h, 2 * k * h - h * dim])
                rigidBody.motion_type = dyno.BodyType.Static
                boxAt = rigid.add_box(box, rigidBody)

    mapper = dyno.DiscreteElementsToTriangleSet3f()
    rigid.state_topology().connect(mapper.in_discrete_elements())
    rigid.graphics_pipeline().push_module(mapper)

    sRender = dyno.GLSurfaceVisualModule()
    sRender.set_color(dyno.Color(0.36078, 0.67451, 0.93333))
    sRender.set_alpha(1)
    mapper.out_triangle_set().connect(sRender.in_triangle_set())
    rigid.graphics_pipeline().push_module(sRender)

    wireRender = dyno.GLWireframeVisualModule()
    wireRender.set_color(dyno.Color(0, 0, 0))
    mapper.out_triangle_set().connect(wireRender.in_edge_set())
    rigid.graphics_pipeline().push_module(wireRender)

    return rigid


compound = createCompound(scn)
boxes = createBoxes(scn)

plane = dyno.PlaneModel3f()
scn.add_node(plane)
plane.var_length_x().set_value(5)
plane.var_length_z().set_value(5)

convoy = dyno.MultibodySystem3f()
scn.add_node(convoy)
boxes.connect(convoy.import_vehicles())
compound.connect(convoy.import_vehicles())

plane.state_triangle_set().connect(convoy.in_triangle_set())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
