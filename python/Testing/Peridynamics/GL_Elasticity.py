import PyPeridyno as dyno

scn = dyno.SceneGraph()

cube = dyno.CubeModel3f()
scn.add_node(cube)
cube.var_location().set_value(dyno.Vector3f([0.6, 0.2, 0.5]))
cube.var_length().set_value(dyno.Vector3f([0.1, 0.1, 0.1]))
cube.graphics_pipeline().disable()

sampler = dyno.ShapeSampler3f()
scn.add_node(sampler)
sampler.var_sampling_distance().set_value(0.005)
sampler.graphics_pipeline().disable()

cube.connect(sampler.import_shape())

initialParticles = dyno.MakeParticleSystem3f()
scn.add_node(initialParticles)

sampler.state_point_set().promote_output().connect(initialParticles.in_points())

bunny = dyno.ElasticBody3f()
scn.add_node(bunny)

initialParticles.connect(bunny.import_solid_particles())

cubeBoundary = dyno.CubeModel3f()
scn.add_node(cubeBoundary)
cubeBoundary.var_location().set_value(dyno.Vector3f([0.5, 0.5, 0.5]))
cubeBoundary.var_length().set_value(dyno.Vector3f([1, 1, 1]))
cubeBoundary.set_visible(False)

cube2vol = dyno.BasicShapeToVolume3f()
scn.add_node(cube2vol)
cube2vol.var_grid_spacing().set_value(0.02)
cube2vol.var_inerted().set_value(True)
cubeBoundary.connect(cube2vol.import_shape())

container = dyno.VolumeBoundary3f()
scn.add_node(container)
cube2vol.connect(container.import_volumes())

bunny.connect(container.import_particle_systems())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
