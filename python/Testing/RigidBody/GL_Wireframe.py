import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

rigidBody = dyno.RigidBodyInfo()
rigidBody.linear_velocity = dyno.Vector3f([1,0,0])
box = dyno.BoxInfo()
box.half_length = dyno.Vector3f([0.5,0.5,0.5]) * dyno.Vector3f([0.065,0.065,0.1])

for i in range(8, 1, -1):
    for j in range(i + 1):
        rigidBody.position= dyno.Vector3f([0.5, 0.5, 0.5]) * dyno.Vector3f(
            [0.5, 1.1 - 0.13 * i, 0.12 + 0.21 * j + 0.1 * (8 - i)])
        rigid.add_box(box, rigidBody)

sphere = dyno.SphereInfo()
sphere.center = dyno.Vector3f([0.5, 0.75,0.5])
sphere.radius = 0.025

rigidSphere = dyno.RigidBodyInfo()
rigid.add_sphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.95, 0.5])
sphere.radius = 0.025
rigid.add_sphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.65, 0.5])
sphere.radius = 0.05
rigid.add_sphere(sphere, rigidSphere)

tet = dyno.TetInfo()
rigidTet = dyno.RigidBodyInfo()
tet.v = [
    dyno.Vector3f([0.5, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.2, 0.5]),
    dyno.Vector3f([0.6, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.1, 0.6]),
]
rigidSphere.position = dyno.Vector3f([0,0,0])
rigid.add_tet(tet, rigidTet)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(1,1,0))
sRender.set_alpha(1.0)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
rigid.graphics_pipeline().push_module(sRender)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
