import PyPeridyno as dyno

scene = dyno.SceneGraph()

root = dyno.OceanPatch3f()
root.var_wind_type().set_value(8)

mapper = dyno.HeightFieldToTriangleSet3f()
root.state_height_field().connect(mapper.in_height_field())
root.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0, 0.2, 1.0))
sRender.var_use_vertex_normal().connect(sRender.in_triangle_set())
mapper.out_triangle_set().connect(sRender.in_triangle_set())
root.graphics_pipeline().push_module(sRender)

scene.add_node(root)
