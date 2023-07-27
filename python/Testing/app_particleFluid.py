import PyPeridyno as dyno
import os

scn = dyno.SceneGraph()

scn.set_upper_bound(dyno.Vector3f([1.5, 1, 1.5]))
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))

boundary = dyno.StaticBoundary3f()
boundary.load_cube(dyno.Vector3f([-0.5, 0, -0.5]), dyno.Vector3f([1.5, 2, 1.5]), 0.02, True)

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../data/bowl/bowl.sdf"
file_path = os.path.join(script_dir, relative_path)
if os.path.isfile(file_path):
    boundary.load_sdf(file_path, False)
else:
    print(f"File not found: {file_path}")
scn.add_node(boundary)

fluid = dyno.ParticleFluid3f()
fluid.load_particles(dyno.Vector3f([0.5, 0.2, 0.4]), dyno.Vector3f([0.7, 1.5, 0.6]), 0.005)
scn.add_node(fluid)
#E:\peridyno-python\src\Rendering\GUI\ImWidgets

calcNorm = dyno.CalculateNorm3f()
fluid.state_velocity().connect(calcNorm.in_vec())
fluid.graphics_pipeline().push_module(calcNorm)


colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5.0)
calcNorm.out_norm().connect(colorMapper.in_scalar())
fluid.graphics_pipeline().push_module(colorMapper)

ptRender = dyno.GLPointVisualModule3f()
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_colorMapMode(ptRender.ColorMapMode.PER_VERTEX_SHADER)
#ptRender.set_colorMapRange(0, 5)

fluid.current_topology().connect(ptRender.in_pointSet())
colorMapper.out_color().connect(ptRender.in_color())

fluid.graphics_pipeline().push_module(ptRender)

colorBar = dyno.ImColorbar3f()
colorBar.var_max().set_value(5)
calcNorm.out_norm().connect(colorBar.in_scalar())
fluid.graphics_pipeline().push_module(colorBar)

fluid.connect(boundary.import_particle_systems())

app = dyno.GLApp()
app.set_scenegraph(scn)
app.initialize(800, 600,True)
app.main_loop()
