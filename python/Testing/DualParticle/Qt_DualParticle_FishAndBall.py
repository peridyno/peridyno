import PyPeridyno as dyno
from PyPeridyno import Vector3f

scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([3, 3, 3]))
scn.set_lower_bound(dyno.Vector3f([-3, -3, -3]))

ptsLoader = dyno.PointsLoader3f()
scn.add_node(ptsLoader)
ptsLoader.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "fish/FishPoints.obj"))
ptsLoader.var_rotation().set_value(dyno.Vector3f([0, 3.14 * 2 / 5, 0]))
ptsLoader.var_location().set_value(dyno.Vector3f([0, 0.4, 0.3]))
initialParticles = dyno.MakeParticleSystem3f()
scn.add_node(initialParticles)
ptsLoader.out_point_set().promote_output().connect(initialParticles.in_points())

fluid = dyno.DualParticleFluid3f()
scn.add_node(fluid)
fluid.var_reshuffle_particles().set_value(True)
initialParticles.connect(fluid.import_initial_states())

ball = dyno.SphereModel3f()
scn.add_node(ball)
ball.var_scale().set_value(Vector3f([0.38, 0.38, 0.38]))
ball.var_location().set_value(Vector3f([0, 0, 0.3]))
sRenderf = dyno.GLSurfaceVisualModule()
sRenderf.set_color(dyno.Color(0.8, 0.52, 0.25))
sRenderf.set_visible(True)
sRenderf.var_use_vertex_normal().set_value(True)
ball.state_triangle_set().connect(sRenderf.in_triangle_set())
ball.graphics_pipeline().push_module(sRenderf)

pm_collide = dyno.TriangularMeshBoundary3f()
scn.add_node(pm_collide)
ball.state_triangle_set().connect(pm_collide.in_triangle_set())
fluid.connect(pm_collide.import_particle_systems())

calculateNorm = dyno.CalculateNorm3f()
fluid.state_velocity().connect(calculateNorm.in_vec())
fluid.graphics_pipeline().push_module(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5.0)
calculateNorm.out_norm().connect(colorMapper.in_scalar())
fluid.graphics_pipeline().push_module(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(1, 0, 0))

ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)
fluid.state_point_set().connect(ptRender.in_point_set())
colorMapper.out_color().connect(ptRender.in_color())
fluid.graphics_pipeline().push_module(ptRender)

# A simple color bar widget for node
colorBar = dyno.ImColorbar3f()
colorBar.var_max().set_value(5.0)
colorBar.var_field_name().set_value("Velocity")
calculateNorm.out_norm().connect(colorBar.in_scalar())
# add the widget to app
fluid.graphics_pipeline().push_module(colorBar)

vpRender = dyno.GLPointVisualModule()
vpRender.set_color(dyno.Color(1, 1, 0))
vpRender.set_color_map_mode(vpRender.ColorMapMode.PER_VERTEX_SHADER)
fluid.state_virtual_pointSet().connect(vpRender.in_point_set())
vpRender.var_point_size().set_value(0.0005)
fluid.graphics_pipeline().push_module(vpRender)


app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
