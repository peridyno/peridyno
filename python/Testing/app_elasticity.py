import PyPeridyno as dyno

scn = dyno.SceneGraph()

emitter = dyno.ParticleEmitterSquare3f()
emitter.var_location().set_value(dyno.Vector3f([0.5, 0.5, 0.5]))
scn.add_node(emitter)

fluid = dyno.ParticleFluid3f()
fluid.load_particles(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([0.2, 0.2, 0.2]), 0.005)
scn.add_node(fluid)

boundary = dyno.StaticBoundary3f()
boundary.load_cube(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([1.0, 1.0, 1.0]), 0.02, True)
scn.add_node(boundary)

calcNorm = dyno.CalculateNorm3f()
colorMapper = dyno.ColorMapping3f()

pointRender = dyno.GLPointVisualModule3f()
pointRender.set_color(dyno.Vector3f([1, 0, 0]))
pointRender.set_colorMapMode(pointRender.ColorMapMode.PER_VERTEX_SHADER)
pointRender.set_colorMapRange(0, 5)

fluid.state_velocity().connect(calcNorm.in_vec())
fluid.current_topology().connect(pointRender.in_pointSet())
calcNorm.out_norm().connect(colorMapper.in_scalar())
colorMapper.out_color().connect(pointRender.in_color())

fluid.graphics_pipeline().push_module(calcNorm)
fluid.graphics_pipeline().push_module(colorMapper)
fluid.graphics_pipeline().push_module(pointRender)

emitter.connect(fluid.import_particles_emitters())
fluid.connect(boundary.import_particle_systems())

app = dyno.GLApp()
app.set_scenegraph(scn)
app.create_window(800, 600)
app.main_loop()