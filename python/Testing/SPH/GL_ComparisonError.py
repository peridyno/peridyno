import PyPeridyno as dyno

PARTICLE_SPACING = 0.005
SMOOTHINGLEHGTH = PARTICLE_SPACING*2.0
TIME_STEP_SIZE = 0.001

scn = dyno.SceneGraph()
scn.set_total_time(3)
scn.set_gravity(dyno.Vector3f([0, 0, 0]))
scn.set_lower_bound(dyno.Vector3f([-2,-2,-2]))
scn.set_upper_bound(dyno.Vector3f([2,2,2]))

cube = dyno.CubeModel3f()
scn.add_node(cube)
cube.var_location().set_value(dyno.Vector3f([0,0,0]))
cube.var_scale().set_value(dyno.Vector3f([0.02,0.5,0.5]))

sampler = dyno.ShapeSampler3f()
scn.add_node(sampler)
sampler.var_sampling_distance().set_value(PARTICLE_SPACING)
sampler.set_visible(False)

cube.connect(sampler.import_shape())

initialParticles = dyno.MakeParticleSystem3f()
scn.add_node(initialParticles)
initialParticles.var_initial_velocity().set_value(dyno.Vector3f([0,0,0]))

sampler.state_point_set().promote_output().connect(initialParticles.in_points())

sphere2 = dyno.SphereModel3f()
scn.add_node(sphere2)
sphere2.var_location().set_value(dyno.Vector3f([-0.3,0,0]))
sphere2.var_radius().set_value(0.05)

sampler2 = dyno.ShapeSampler3f()
scn.add_node(sampler2)
sampler2.var_sampling_distance().set_value(PARTICLE_SPACING)
sampler.set_visible(False)

sphere2.connect(sampler2.import_shape())

initialParticles2 = dyno.MakeParticleSystem3f()
scn.add_node(initialParticles2)
initialParticles2.var_initial_velocity().set_value(dyno.Vector3f([1,0,0]))

sampler2.state_point_set().promote_output().connect(initialParticles2.in_points())

fluid = dyno.ParticleFluid3f()
scn.add_node(fluid)
fluid.set_dt(TIME_STEP_SIZE)
fluid.var_reshuffle_particles().set_value(True)
initialParticles.connect(fluid.import_initial_states())
initialParticles2.connect(fluid.import_initial_states())

fluid.animation_pipeline().clear()
smoothingLength = fluid.animation_pipeline().create_modules()
smoothingLength.var_value().set_value(SMOOTHINGLEHGTH)

samplingDistance = fluid.animation_pipeline().create_modules()
samplingDistance.var_value().set_value(PARTICLE_SPACING)

integrator = dyno.ParticleIntegrator3f()
fluid.state_time_step().connect(integrator.in_time_step())
fluid.state_position().connect(integrator.in_position())
fluid.state_velocity().connect(integrator.in_velocity())
fluid.animation_pipeline().push_module(integrator)

nbrQuery = dyno.NeighborPointQuery3f()
smoothingLength.out_floating().connect(nbrQuery.in_radius())
fluid.state_position().connect(nbrQuery.in_position())
fluid.animation_pipeline().push_module(nbrQuery)

density = dyno.ImplicitISPH3f()
density.var_iteration_number().set_value(10)
smoothingLength.out_floating().connect(density.in_smoothing_length())
samplingDistance.out_floating().connect(density.in_sampling_distance())
fluid.state_time_step().connect(density.in_time_step())
fluid.state_position().connect(density.in_position())
fluid.state_velocity().connect(density.in_velocity())
nbrQuery.out_neighbor_ids().connect(density.in_neighbor_ids())
fluid.animation_pipeline().push_module(density)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
