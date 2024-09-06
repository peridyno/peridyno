import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()

rigidBody = dyno.RigidBodyInfo()
rigidBody.linear_velocity = dyno.Vector3f([0.5, 0, 0])

rigidBody.collisionMask = dyno.CollisionMask.CT_BoxOnly
box = dyno.BoxInfo()
for i in range(8, 1, -1):
    for j in range(i + 1):
        box.center = dyno.Vector3f([0.5, 0.5, 0.5]) * dyno.Vector3f(
            [0.5, 1.1 - 0.13 * i, 0.12 + 0.21 * j + 0.1 * (8 - i)])
        box.half_length = dyno.Vector3f([0.5, 0.5, 0.5]) * dyno.Vector3f([0.065, 0.065, 0.1])
        rigid.add_box(box, rigidBody)

sphere = dyno.SphereInfo()
sphere.center = dyno.Vector3f([0.5, 0.75, 0.5])
sphere.radius = 0.025

rigidSphere = dyno.RigidBodyInfo()
rigidSphere.collisionMask = dyno.CollisionMask.CT_SphereOnly
rigid.add_sphere(sphere, rigidSphere)

sphere.center = dyno.Vector3f([0.5, 0.95, 0.5])
sphere.radius = 0.025
rigid.add_sphere(sphere, rigidSphere)

sphere.center = dyno.Vector3f([0.5, 0.65, 0.5])
sphere.radius = 0.05
rigid.add_sphere(sphere, rigidSphere)

tet = dyno.TetInfo()
tet.v = [
    dyno.Vector3f([0.5, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.2, 0.5]),
    dyno.Vector3f([0.6, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.1, 0.6]),
]
rigid.add_tet(tet, rigidSphere)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(1, 1, 0))
sRender.set_alpha(0.5)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
rigid.graphics_pipeline().push_module(sRender)

scn.add_node(rigid)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
