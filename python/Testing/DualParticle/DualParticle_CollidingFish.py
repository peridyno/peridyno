import os

import PyPeridyno as dyno


def filePath(str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([3, 3, 3]))
scn.set_lower_bound(dyno.Vector3f([-3, -3, -3]))
scn.set_gravity(dyno.Vector3f([0, 0, 0]))

ptsLoader = dyno.PointsLoader3f()
ptsLoader.var_file_name().set_value(dyno.FilePath(filePath("fish/FishPoints.obj")))
ptsLoader.var_rotation().set_value(dyno.Vector3f([0, 0, 3.1415926]))
ptsLoader.var_location().set_value(dyno.Vector3f([0, 0, 0.23]))
initialParticles = dyno.MakeParticleSystem3f()
initialParticles.var_initial_velocity().set_value(dyno.Vector3f([0, 0, -1.5]))
ptsLoader.out_point_set().promote_output().connect(initialParticles.in_points())

ptsLoader2 = dyno.PointsLoader3f()
ptsLoader2.var_file_name().set_value(dyno.FilePath(filePath("fish/FishPoints.obj")))
ptsLoader2.var_rotation().set_value(dyno.Vector3f([0, 0, 0]))
ptsLoader2.var_location().set_value(dyno.Vector3f([0, 0, -0.23]))
initialParticles2 = dyno.MakeParticleSystem3f()
initialParticles2.var_initial_velocity().set_value(dyno.Vector3f([0, 0, 1.5]))
ptsLoader2.out_point_set().promote_output().connect(initialParticles2.in_points())

fluid = dyno.DualParticleFluidSystem3f()
fluid.var_reshuffle_particles().set_value(True)
initialParticles.connect(fluid.import_initial_states())
initialParticles2.connect(fluid.import_initial_states())

# Create a boundary
boundary = dyno.StaticBoundary3f()

fluid.connect(boundary.import_particle_systems())

calculateNorm = dyno.CalculateNorm3f()
fluid.state_velocity().connect(calculateNorm.in_vec())
fluid.graphics_pipeline().push_module(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5.0)
calculateNorm.out_norm().connect(colorMapper.in_scalar())
fluid.graphics_pipeline().push_module(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.var_point_size().set_value(0.0035)
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
vpRender.set_color(dyno.Color(1,1,0))
vpRender.set_color_map_mode(vpRender.ColorMapMode.PER_VERTEX_SHADER)
fluid.state_virtual_pointSet().connect(vpRender.in_point_set())
vpRender.var_point_size().set_value(0.0005)
fluid.graphics_pipeline().push_module(vpRender)

# scn.add_node(ptsLoader)
scn.add_node(initialParticles)
# scn.add_node(ptsLoader2)
scn.add_node(initialParticles2)
scn.add_node(fluid)
scn.add_node(boundary)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
