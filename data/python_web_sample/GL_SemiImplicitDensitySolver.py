import PyPeridyno as dyno

scene = dyno.SceneGraph()
scene.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))
scene.set_upper_bound(dyno.Vector3f([0.5, 1, 0.5]))

cube = dyno.CubeModel3f()
cube.var_location().set_value(dyno.Vector3f([0, 0.2, 0]))
cube.var_length().set_value(dyno.Vector3f([0.2, 0.2, 0.2]))
cube.graphics_pipeline().disable()

sampler = dyno.CubeSampler3f()
sampler.var_sampling_distance().set_value(0.005)
sampler.graphics_pipeline().disable()

cube.out_cube().connect(sampler.in_cube())

initialParticles = dyno.MakeParticleSystem3f()

sampler.state_point_set().promote_output().connect(initialParticles.in_points())

fluid = dyno.ParticleFluid3f()
initialParticles.connect(fluid.import_initial_states())

boundary = dyno.StaticBoundary3f()
boundary.load_cube(dyno.Vector3f([-0.5, 0, -0.5]), dyno.Vector3f([0.5, 1, 0.5]), 0.02, True)
fluid.connect(boundary.import_particle_systems())

calculateNorm = dyno.CalculateNorm3f()
fluid.state_velocity().connect(calculateNorm.in_vec())
fluid.graphics_pipeline().push_module(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5)
calculateNorm.out_norm().connect(colorMapper.in_scalar())
fluid.graphics_pipeline().push_module(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

fluid.state_point_set().connect(ptRender.in_point_set())
colorMapper.out_color().connect(ptRender.in_color())

fluid.graphics_pipeline().push_module(ptRender)

colorBar = dyno.ImColorbar3f()
colorBar.var_max().set_value(5)
colorBar.var_field_name().set_value("Velocity")
calculateNorm.out_norm().connect(colorBar.in_scalar())

fluid.graphics_pipeline().push_module(colorBar)

scene.add_node(cube)
scene.add_node(sampler)
scene.add_node(initialParticles)
scene.add_node(fluid)
scene.add_node(boundary)
