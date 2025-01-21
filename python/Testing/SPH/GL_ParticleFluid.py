import PyPeridyno as dyno

scn = dyno.SceneGraph()

scn.set_upper_bound(dyno.Vector3f([1.5, 1, 1.5]))
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))

cube = dyno.CubeModel3f()
scn.add_node(cube)
cube.var_location().set_value(dyno.Vector3f([0.6, 0.5, 0.5]))
cube.var_length().set_value(dyno.Vector3f([0.5,0.5,0.5]))
cube.graphics_pipeline().disable()

sampler = dyno.ShapeSampler3f()
scn.add_node(sampler)
sampler.var_sampling_distance().set_value(0.005)
sampler.graphics_pipeline().disable()

cube.connect(sampler.import_shape())

initialParticles = dyno.MakeParticleSystem3f()
scn.add_node(initialParticles)

sampler.state_point_set().promote_output().connect(initialParticles.in_points())

fluid = dyno.ParticleFluid3f()
scn.add_node(fluid)
fluid.var_reshuffle_particles().set_value(True)
initialParticles.connect(fluid.import_initial_states())

volLoader = dyno.VolumeLoader3f()
scn.add_node(volLoader)
volLoader.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "bowl/bowl.sdf"))

volBoundary = dyno.VolumeBoundary3f()
scn.add_node(volBoundary)
volLoader.connect(volBoundary.import_volumes())

fluid.connect(volBoundary.import_particle_systems())

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
cube2vol.connect(volBoundary.import_volumes())

staticMesh = dyno.StaticMeshLoader3f()
scn.add_node(staticMesh)
staticMesh.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() +"bowl/bowl.obj" ))

fluid.graphics_pipeline().clear()

calculateNorm = dyno.CalculateNorm3f()
fluid.state_velocity().connect(calculateNorm.in_vec())
fluid.graphics_pipeline().push_module(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5)
calculateNorm.out_norm().connect(colorMapper.in_scalar())
fluid.graphics_pipeline().push_module(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(1,0,0))
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
app.initialize(800, 600, True)
app.main_loop()
