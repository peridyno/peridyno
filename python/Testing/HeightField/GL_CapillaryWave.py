import PyPeridyno as dyno

scn = dyno.SceneGraph()

root = dyno.CapillaryWave3f()

mapper = dyno.HeightFieldToTriangleSet3f()

root.state_height_field().connect(mapper.in_height_field())
root.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0, 0.2, 1.0))
mapper.out_triangle_set().connect(sRender.in_triangle_set())
root.graphics_pipeline().push_module(sRender)

scn.add_node(root)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.render_window().get_camera().set_unit_scale(52)
app.main_loop()
