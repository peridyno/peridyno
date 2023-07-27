import PyPeridyno as dyno

scn = dyno.SceneGraph()
paticle = dyno.ParticleSystem3f()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

rigidBody = dyno.RigidBodyInfo()

rigidBody.linear_velocity = dyno.Vector3f([0.5, 0, 0])
box = dyno.BoxInfo()
for i in range(8, 1, -1):
    for j in range(i + 1):
        box.center = dyno.Vector3f([0.5 * 0.5, 0.5 * (1.1 - 0.13 * i), 0.5 * (0.12 + 0.21 * j + 0.1 * (8 - i))])
        box.half_length = dyno.Vector3f([0.5 * 0.065, 0.5 * 0.065, 0.5 * 0.1])
        rigid.add_box(box, rigidBody, 1)

sphere = dyno.SphereInfo()
sphere.center = dyno.Vector3f([0.5, 0.75, 0.5])
sphere.radius = 0.025

rigidSphere = dyno.RigidBodyInfo() 
rigid.add_sphere(sphere, rigidSphere, 1)

sphere.center = dyno.Vector3f([0.5, 0.95, 0.5])
sphere.radius = 0.025
rigid.add_sphere(sphere, rigidSphere, 1)

sphere.center = dyno.Vector3f([0.5, 0.65, 0.5])
sphere.radius = 0.05
rigid.add_sphere(sphere, rigidSphere, 1)


tet = dyno.TetInfo()

tet.v = [
    dyno.Vector3f([0.5, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.2, 0.5]),
    dyno.Vector3f([0.6, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.1, 0.6]),
]
rigid.add_tet(tet, rigidSphere, 1)


mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.current_topology().connect(mapper.in_discreteElements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule3f()
sRender.set_color(dyno.Color(1, 1, 0))
mapper.out_triangleSet().connect(sRender.in_triangleSet())
rigid.graphics_pipeline().push_module(sRender)

#todo:
elementQuery = dyno.NeighborElementQuery3f()
rigid.current_topology().connect(elementQuery.in_discreteElements())
rigid.state_collisionMask().connect(elementQuery.in_collisionMask())
rigid.graphics_pipeline().push_module(elementQuery)

contactMapper = dyno.ContactsToEdgeSet3f()
elementQuery.out_contacts().connect(contactMapper.in_contacts())
contactMapper.var_scale().setValue(0.02)
rigid.graphics_pipeline().push_module(contactMapper)


wireRender = dyno.GLWireframeVisualModule() 
wireRender.set_color(dyno.Color(0, 1, 0))
contactMapper.out_edge_set().connect(wireRender.in_edgeSet()) 
rigid.graphics_pipeline().push_module(wireRender) 



pointRender = dyno.GLPointVisualModule() 
pointRender.set_color(dyno.Color(1, 0, 0))
contactPointMapper.out_pointSet().connect(pointRender.in_pointSet()) 
rigid.graphics_pipeline().push_module(pointRender) 

app = dyno.GLApp()
app.set_scenegraph(scn)
app.initialize(1280, 768, True)
app.main_loop()