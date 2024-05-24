import PyPeridyno as dyno

scene = dyno.SceneGraph()
scene.set_total_time(3)
scene.set_gravity(dyno.Vector3f([0, -9.8, 0]))
scene.set_lower_bound(dyno.Vector3f([-1, 0, 0]))
scene.set_upper_bound(dyno.Vector3f([1, 1, 1]))

emitter = dyno.SquareEmitter3f()
emitter.var_location().set_value(dyno.Vector3f([0, 0.5, 0.5]))

fluid = dyno.ParticleFluid3f()
emitter.connect(fluid.import_particle_emitters())

ptRender = dyno.GLPointVisualModule()
ptRender.var_point_size().set_value(0.002)
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

calculateNorm = dyno.CalculateNorm3f()
colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5)
fluid.state_velocity().connect(calculateNorm.in_vec())
calculateNorm.out_norm().connect(colorMapper.in_scalar())

colorMapper.out_color().connect(ptRender.in_color())
fluid.state_point_set().connect(ptRender.in_point_set())

fluid.graphics_pipeline().push_module(calculateNorm)
fluid.graphics_pipeline().push_module(colorMapper)
fluid.graphics_pipeline().push_module(ptRender)

barricade = dyno.StaticTriangularMesh3f()
barricade.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "bowl/barricade.obj"))
barricade.var_location().set_value(dyno.Vector3f([0.1, 0.022, 0.5]))

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0.8, 0.52, 0.25))
sRender.set_visible(True)
sRender.var_use_vertex_normal().set_value(True)
barricade.state_triangle_set().connect(sRender.in_triangle_set())
barricade.graphics_pipeline().push_module(sRender)

boundary = dyno.StaticTriangularMesh3f()
boundary.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "standard/standard_cube2.obj"))
boundary.graphics_pipeline().disable()

sfi = dyno.TriangularMeshBoundary3f()
pbd = dyno.SemiAnalyticalPositionBasedFluidModel3f()
pbd.var_smoothing_length().set_value(0.0085)

merge = dyno.MergeTriangleSet3f()
boundary.state_triangle_set().connect(merge.in_first())
barricade.state_triangle_set().connect(merge.in_second())

fluid.connect(sfi.import_particle_systems())
merge.state_triangle_set().connect(sfi.in_triangle_set())

scene.add_node(emitter)
scene.add_node(fluid)
scene.add_node(barricade)
scene.add_node(boundary)
scene.add_node(sfi)
scene.add_node(merge)
