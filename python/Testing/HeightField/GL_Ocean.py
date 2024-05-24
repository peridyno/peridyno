import PyPeridyno as dyno

scn = dyno.SceneGraph()

oceanPatch = dyno.OceanPatch3f()
oceanPatch.var_wind_type().set_value(8)

root = dyno.Ocean3f()
root.var_extentX().set_value(2)
root.var_extentZ().set_value(2)
oceanPatch.connect(root.import_ocean_patch())

waves = dyno.CapillaryWave3f()
waves.connect(root.import_capillary_waves())

mapper = dyno.HeightFieldToTriangleSet3f()

root.state_height_field().connect(mapper.in_height_field())
root.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0, 0.2, 1.0))
sRender.var_use_vertex_normal().set_value(True)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
root.graphics_pipeline().push_module(sRender)

scn.add_node(oceanPatch)
scn.add_node(root)
scn.add_node(waves)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.render_window().get_camera().set_unit_scale(52)
app.main_loop()
