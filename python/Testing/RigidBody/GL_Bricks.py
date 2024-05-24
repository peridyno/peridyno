import os

import PyPeridyno as dyno


def filePath(str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()

rigidBody = dyno.RigidBodyInfo()
rigidBody.linear_velocity = dyno.Vector3f([0.5, 0, 0])
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

scn.add_node(rigid)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
