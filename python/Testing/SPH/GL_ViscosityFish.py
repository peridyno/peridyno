import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([1.5, 1, 1.5]))
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))

ptsLoader = dyno.PointsLoader3f()
scn.add_node(ptsLoader)
ptsLoader.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "fish/FishPoints.obj"))
ptsLoader.var_rotation().set_value(dyno.Vector3f([0, 0, 3.1415926]))
ptsLoader.var_location().set_value(dyno.Vector3f([0, 0.15, 0.23]))
initialParticles = dyno.MakeParticleSystem3f()
scn.add_node(initialParticles)
ptsLoader.out_point_set().promote_output().connect(initialParticles.in_points())

fluid = dyno.ParticleFluid3f()
scn.add_node(fluid)
fluid.var_reshuffle_particles().set_value(True)
initialParticles.connect(fluid.import_initial_states())

fluid.animation_pipeline().clear()

smoothingLength = dyno.FloatingNumber3f()
fluid.animation_pipeline().push_module(smoothingLength)
smoothingLength.var_value().set_value(0.0125)

integrator = dyno.ParticleIntegrator3f()
fluid.state_time_step().connect(integrator.in_time_step())
fluid.state_position().connect(integrator.in_position())
fluid.state_velocity().connect(integrator.in_velocity())
fluid.animation_pipeline().push_module(integrator)

nbrQuery = dyno.NeighborPointQuery3f()
smoothingLength.out_floating().connect(nbrQuery.in_radius())
fluid.state_position().connect(nbrQuery.in_position())
fluid.animation_pipeline().push_module(nbrQuery)

simple = dyno.SimpleVelocityConstraint3f()
simple.var_viscosity().set_value(500)
simple.var_simple_iteration_enable().set_value(False)
fluid.state_time_step().connect(simple.in_time_step())
smoothingLength.out_floating().connect(simple.in_smoothing_length())
fluid.state_position().connect(simple.in_position())
fluid.state_velocity().connect(simple.in_velocity())

simple.in_sampling_distance().set_value(0.005)
nbrQuery.out_neighbor_ids().connect(simple.in_neighbor_ids())
fluid.animation_pipeline().push_module(simple)

cubeBoundary = dyno.CubeModel3f()
scn.add_node(cubeBoundary)
cubeBoundary.var_location().set_value(dyno.Vector3f([0.5,1,0.5]))
cubeBoundary.var_length().set_value(dyno.Vector3f([2,2,2]))
cubeBoundary.set_visible(False)

cube2vol = dyno.BasicShapeToVolume3f()
scn.add_node(cube2vol)
cube2vol.var_grid_spacing().set_value(0.02)
cube2vol.var_inerted().set_value(True)
cubeBoundary.connect(cube2vol.import_shape())

container = dyno.VolumeBoundary3f()
scn.add_node(container)
cube2vol.connect(container.import_volumes())
fluid.connect(container.import_particle_systems())

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

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
