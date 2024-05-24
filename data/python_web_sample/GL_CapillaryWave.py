import PyPeridyno as dyno

scene = dyno.SceneGraph()

root = dyno.CapillaryWave3f()

mapper = dyno.HeightFieldToTriangleSet3f()

root.state_height_field().connect(mapper.in_height_field())
root.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0, 0.2, 1.0))
mapper.out_triangle_set().connect(sRender.in_triangle_set())
root.graphics_pipeline().push_module(sRender)

scene.add_node(root)
