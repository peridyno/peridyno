import PyPeridyno as dyno

scene = dyno.SceneGraph()

emitter = dyno.SquareEmitter3f()
emitter.var_location().set_value(dyno.Vector3f([0.5, 0.5, 0.5]))

fluid = dyno.ParticleFluid3f()
fluid.load_particles(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([0.2, 0.2, 0.2]), 0.05)

emitter.connect(fluid.import_particle_emitters())

calculateNorm = dyno.CalculateNorm3f()
colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(0.5)

ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

fluid.state_velocity().connect(calculateNorm.in_vec())
fluid.state_point_set().connect(ptRender.in_point_set())
calculateNorm.out_norm().connect(colorMapper.in_scalar())
colorMapper.out_color().connect(ptRender.in_color())

fluid.graphics_pipeline().push_module(calculateNorm)
fluid.graphics_pipeline().push_module(colorMapper)
fluid.graphics_pipeline().push_module(ptRender)

container = dyno.StaticBoundary3f()
container.load_cube(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([1.0, 1.0, 1.0]), 0.02, True)

fluid.connect(container.import_particle_systems())

scene.add_node(emitter)
scene.add_node(fluid)
scene.add_node(container)