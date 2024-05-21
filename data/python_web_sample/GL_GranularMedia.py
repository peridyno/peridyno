import PyPeridyno as dyno

scene = dyno.SceneGraph()

root = dyno.GranularMedia3f()

mapper = dyno.HeightFieldToTriangleSet3f()
root.state_height_field().connect(mapper.in_height_field())
root.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0.8, 0.8, 0.8))
sRender.var_use_vertex_normal().set_value(True)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
root.graphics_pipeline().push_module(sRender)

tracking = dyno.SurfaceParticleTracking3f()
root.connect(tracking.import_granular_media())

scene.add_node(root)
scene.add_node(tracking)
