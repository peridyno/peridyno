import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([10.5,5,10.5]))
scn.set_lower_bound(dyno.Vector3f([-10.5,5,-10.5]))

obj1 = dyno.ObjLoader3f()
scn.add_node(obj1)
obj1.var_scale().set_value(dyno.Vector3f([0.3,0.3,0.3]))
obj1.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "plane/plane_lowRes.obj"))
obj1.var_location().set_value(dyno.Vector3f([0,0,0]))
SurfaceModule1 = obj1.graphics_pipeline().find_first_module_surface()
SurfaceModule1.set_color(dyno.Color(0.2, 0.2, 0.2))
SurfaceModule1.set_metallic(1)
SurfaceModule1.set_roughness(0.8)

pointset_1 = dyno.ParticleRelaxtionOnMesh3f()
scn.add_node(pointset_1)
pointset_1.var_sampling_distance().set_value(0.005)
pointset_1.var_thickness().set_value(0.045)
obj1.out_triangle_set().connect(pointset_1.in_triangle_set())
pointset_1.graphics_pipeline().clear()

ghost2 = dyno.MakeGhostParticles3f()
scn.add_node(ghost2)
pointset_1.state_point_set().connect(ghost2.in_points())
pointset_1.state_point_normal().connect(ghost2.state_normal())

cube = dyno.CubeModel3f()
scn.add_node(cube)
cube.var_location().set_value(dyno.Vector3f([0,0.3,0]))
cube.var_length().set_value(dyno.Vector3f([0.2,0.2,0.2]))
cube.set_visible(False)

sampler = dyno.ShapeSampler3f()
scn.add_node(sampler)
sampler.var_sampling_distance().set_value(0.005)
sampler.set_visible(False)

cube.connect(sampler.import_shape())

fluidParticles = dyno.MakeParticleSystem3f()
scn.add_node(fluidParticles)

sampler.state_point_set().promote_output().connect(fluidParticles.in_points())

incompressibleFluid = dyno.GhostFluid3f()
scn.add_node(incompressibleFluid)
fluidParticles.connect(incompressibleFluid.import_initial_states())
ghost2.connect(incompressibleFluid.import_boundary_particles())


ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(0.6,0.5,0.2))
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)
pointset_1.state_point_set().connect(ptRender.in_point_set())
pointset_1.graphics_pipeline().push_module(ptRender)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
