import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([3, 3, 3]))
scn.set_lower_bound(dyno.Vector3f([-3, -3, -3]))

emitter = dyno.PoissonEmitter3f()
scn.add_node(emitter)
emitter.var_rotation().set_value(dyno.Vector3f([0, 0, -90]))
emitter.var_sampling_distance().set_value(0.008)
emitter.var_emitter_shape().get_data_ptr().set_current_key(1)
emitter.var_width().set_value(0.1)
emitter.var_height().set_value(0.1)
emitter.var_velocity_magnitude().set_value(1.5)
emitter.var_location().set_value(dyno.Vector3f([0.2, 0.5, 0]))

emitter2 = dyno.PoissonEmitter3f()
scn.add_node(emitter2)
emitter2.var_rotation().set_value(dyno.Vector3f([0, 0, 90]))
emitter2.var_sampling_distance().set_value(0.008)
emitter2.var_emitter_shape().get_data_ptr().set_current_key(1)
emitter2.var_width().set_value(0.1)
emitter2.var_height().set_value(0.1)
emitter2.var_velocity_magnitude().set_value(1.5)
emitter2.var_location().set_value(dyno.Vector3f([-0.2, 0.5, -0]))

temp = dyno.DualParticleFluid3f()
fluid = dyno.DualParticleFluid3f(temp.EVirtualParticleSamplingStrategy.SpatiallyAdaptiveStrategy)
scn.add_node(fluid)

emitter.connect(fluid.import_particle_emitters())
emitter2.connect(fluid.import_particle_emitters())

calculateNorm = dyno.CalculateNorm3f()
fluid.state_velocity().connect(calculateNorm.in_vec())
fluid.graphics_pipeline().push_module(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5.0)
calculateNorm.out_norm().connect(colorMapper.in_scalar())
fluid.graphics_pipeline().push_module(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.var_point_size().set_value(0.0035)
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)
fluid.state_point_set().connect(ptRender.in_point_set())
colorMapper.out_color().connect(ptRender.in_color())
fluid.graphics_pipeline().push_module(ptRender)

# A simple color bar widget for node
colorBar = dyno.ImColorbar3f()
colorBar.var_max().set_value(5.0)
colorBar.var_field_name().set_value("Velocity")
calculateNorm.out_norm().connect(colorBar.in_scalar())
# add the widget to app
fluid.graphics_pipeline().push_module(colorBar)

vpRender = dyno.GLPointVisualModule()
vpRender.set_color(dyno.Color(1, 1, 0))
vpRender.set_color_map_mode(vpRender.ColorMapMode.PER_VERTEX_SHADER)
fluid.state_virtual_pointSet().connect(vpRender.in_point_set())
vpRender.var_point_size().set_value(0.001)
fluid.graphics_pipeline().push_module(vpRender)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
