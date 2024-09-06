import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_gravity(dyno.Vector3f([0, -9.8, 0]))

emitter = dyno.CircularEmitter3f()
emitter.var_location().set_value(dyno.Vector3f([0, 1, 0]))

fluid = dyno.ParticleFluid3f()
emitter.connect(fluid.import_particle_emitters())

ptRender = dyno.GLPointVisualModule()
ptRender.var_point_size().set_value(0.002)
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

calculateNorm = dyno.CalculateNorm3f()
colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5)
fluid.state_velocity().connect(calculateNorm.in_vec())
calculateNorm.out_norm().connect(colorMapper.in_scalar())

colorMapper.out_color().connect(ptRender.in_color())
fluid.state_point_set().connect(ptRender.in_point_set())

fluid.graphics_pipeline().push_module(calculateNorm)
fluid.graphics_pipeline().push_module(colorMapper)
fluid.graphics_pipeline().push_module(ptRender)

fluid.animation_pipeline().disable()

plane = dyno.PlaneModel3f()
plane.var_scale().set_value(dyno.Vector3f([2, 0, 2]))

sphere = dyno.SphereModel3f()
sphere.var_location().set_value(dyno.Vector3f([0, 0.5, 0]))
sphere.var_scale().set_value(dyno.Vector3f([0.2, 0.2, 0.2]))

merge = dyno.MergeTriangleSet3f()
plane.state_triangle_set().connect(merge.in_first())
sphere.state_triangle_set().connect(merge.in_second())

sfi = dyno.SemiAnalyticalSFINode3f()

fluid.connect(sfi.import_particle_systems())
merge.state_triangle_set().connect(sfi.in_triangle_set())

scn.add_node(emitter)
scn.add_node(fluid)
scn.add_node(plane)
scn.add_node(sphere)
scn.add_node(merge)
scn.add_node(sfi)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
