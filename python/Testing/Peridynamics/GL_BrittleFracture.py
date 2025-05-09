import PyPeridyno as dyno

scn = dyno.SceneGraph()

cube = dyno.CubeModel3f()
scn.add_node(cube)
cube.set_visible(False)
cube.var_location().set_value(dyno.Vector3f([0.2, 0.2, 0]))
cube.var_length().set_value(dyno.Vector3f([0.1, 0.1, 0.1]))
cube.var_segments().set_value(dyno.Vector3i([10, 10, 10]))

sampler = dyno.ShapeSampler3f()
scn.add_node(sampler)
sampler.var_sampling_distance().set_value(0.005)
sampler.graphics_pipeline().disable()

cube.connect(sampler.import_shape())

initialParticles = dyno.MakeParticleSystem3f()
scn.add_node(initialParticles)

sampler.state_point_set().promote_output().connect(initialParticles.in_points())

elastoplasticBody = dyno.ElastoplasticBody3f()
scn.add_node(elastoplasticBody)
initialParticles.connect(elastoplasticBody.import_solid_particles())

elastoplasticBody.animation_pipeline().clear()

integrator = dyno.ParticleIntegrator3f()
elastoplasticBody.state_time_step().connect(integrator.in_time_step())
elastoplasticBody.state_position().connect(integrator.in_position())
elastoplasticBody.state_velocity().connect(integrator.in_velocity())
elastoplasticBody.animation_pipeline().push_module(integrator)

nbrQuery = dyno.NeighborPointQuery3f()
elastoplasticBody.state_horizon().connect(nbrQuery.in_radius())
elastoplasticBody.state_position().connect(nbrQuery.in_position())
elastoplasticBody.animation_pipeline().push_module(nbrQuery)

plasticity = dyno.FractureModule3f()
plasticity.var_cohesion().set_value(0.00001)
elastoplasticBody.state_horizon().connect(plasticity.in_horizon())
elastoplasticBody.state_time_step().connect(plasticity.in_time_step())
elastoplasticBody.state_position().connect(plasticity.in_y())
elastoplasticBody.state_reference_position().connect(plasticity.in_x())
elastoplasticBody.state_velocity().connect(plasticity.in_velocity())
# elastoplasticBody.state_bonds().connect(plasticity.in_bonds())
nbrQuery.out_neighbor_ids().connect(plasticity.in_neighbor_ids())
elastoplasticBody.animation_pipeline().push_module(plasticity)

visModule = dyno.ImplicitViscosity3f()
visModule.var_viscosity().set_value(1)
elastoplasticBody.state_time_step().connect(visModule.in_time_step())
elastoplasticBody.state_horizon().connect(visModule.in_smoothing_length())
elastoplasticBody.state_position().connect(visModule.in_position())
elastoplasticBody.state_velocity().connect(visModule.in_velocity())
nbrQuery.out_neighbor_ids().connect(visModule.in_neighbor_ids())
elastoplasticBody.animation_pipeline().push_module(visModule)

cubeBoundary = dyno.CubeModel3f()
scn.add_node(cubeBoundary)
cubeBoundary.var_location().set_value(dyno.Vector3f([0.5, 1, 0.5]))
cubeBoundary.var_length().set_value(dyno.Vector3f([2, 2, 2]))
cubeBoundary.set_visible(False)

cube2vol = dyno.BasicShapeToVolume3f()
scn.add_node(cube2vol)
cube2vol.var_grid_spacing().set_value(0.02)
cube2vol.var_inerted().set_value(True)
cubeBoundary.connect(cube2vol.import_shape())

container = dyno.VolumeBoundary3f()
scn.add_node(container)
container.var_tangential_friction().set_value(0.95)
cube2vol.connect(container.import_volumes())

elastoplasticBody.connect(container.import_particle_systems())



app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(800, 600, True)
app.main_loop()
