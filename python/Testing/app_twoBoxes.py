import PyPeridyno as dyno

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.add_node(rigid)

rigidBody = dyno.RigidBodyInfo()
rigidBody.linear_velocity = dyno.Vector3f([0, 0, 0])

box = dyno.BoxInfo()
box.center = dyno.Vector3f([0.5, 0.1, 0.5])
box.halfLength = dyno.Vector3f([0.1, 0.1, 0.1])
rigid.add_box(box, rigidBody, 1)

box.center = dyno.Vector3f([0.5, 0.3, 0.59])
box.halfLength = dyno.Vector3f([0.1, 0.1, 0.1])
rigid.add_box(box, rigidBody, 1)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.current_topology().connect(mapper.in_discreteElements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule3f()
sRender.set_color(dyno.Vector3f([1, 1, 0]))
mapper.out_triangleSet().connect(sRender.in_triangleSet())
rigid.graphics_pipeline().push_module(sRender)


app = dyno.GLApp()
app.set_scenegraph(scn)
app.create_window(1280, 768)
app.main_loop()