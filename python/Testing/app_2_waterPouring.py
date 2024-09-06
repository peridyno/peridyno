import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_gravity(dyno.Vector3f([0.0, -9.8, 0.0]))

dyno.ModelingInitializer.modeling_init_static_plugin()
dyno.ParticleSystemInitializer.paticleSystem_init_static_plugin()

# createScene:

# Create a particle emitter
emitter = dyno.CircularEmitter3f()
emitter.var_location().set_value(dyno.Vector3f([0.0, 1.0, 0.0]))

# Particle fluid node
fluid = dyno.ParticleFluid3f()
emitter.connect(fluid.import_particles_emitters())

ptRender = dyno.GLPointVisualModule()
# ptRender.var_point_size().set_value(0.002)
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_colorMapMode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

calculateNorm = dyno.CalculateNorm3f()
colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5.0)
fluid.state_velocity().connect(calculateNorm.in_vec())
calculateNorm.out_norm().connect(colorMapper.in_scalar())

colorMapper.out_color().connect(ptRender.in_color())
fluid.state_point_set().connect(ptRender.in_pointSet())

fluid.graphics_pipeline().push_module(calculateNorm)
fluid.graphics_pipeline().push_module(colorMapper)
fluid.graphics_pipeline().push_module(ptRender)

# fluid.animation_pipeline().disable()

# Setup boundaries
plane = dyno.PlaneModel3f()
plane.var_scale().set_value(dyno.Vector3f([2.0, 0.0, 2.0]))
scn.add_node(plane)

sphere = dyno.SphereModel3f()
sphere.var_location().set_value(dyno.Vector3f([0.0, 0.5, 0.0]))
sphere.var_scale().set_value(dyno.Vector3f([0.2, 0.2, 0.2]))
scn.add_node(sphere)

merge = dyno.MergeTriangleSet3f()
plane.state_triangleSet().connect(merge.in_first())
sphere.state_triangleSet().connect(merge.in_second())
scn.add_node(merge)

# SFI node
sfi = dyno.SemiAnalyticalSFINode3f()
fluid.connect(sfi.import_particle_systems())
merge.state_triangleSet().connect(sfi.in_triangleSet())

app = dyno.GLApp()
app.set_scenegraph(scn)
app.initialize(1280, 768, True)
app.main_loop()
