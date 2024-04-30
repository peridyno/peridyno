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
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))
scn.set_upper_bound(dyno.Vector3f([0.5, 1, 0.5]))

cube = dyno.CubeModel3f()
cube.var_location().set_value(dyno.Vector3f([0,0.2,0]))
cube.var_length().set_value(dyno.Vector3f([0.2,0.2,0.2]))
cube.graphics_pipeline().disable()

sampler = dyno.CubeSampler3f()
sampler.var_sampling_distance().set_value(0.005)
sampler.graphics_pipeline().disable()

cube.out_cube().connect(sampler.in_cube())

initialParticles = dyno.MakeParticleSystem3f()

sampler.state_point_set().promote_output().connect(initialParticles.in_points())

fluid = dyno.ParticleFluid3f()
initialParticles.connect(fluid.import_initial_states())

boundary = dyno.StaticBoundary3f()
boundary.load_cube(dyno.Vector3f([-0.5,0,-0.5]), dyno.Vector3f([0.5,1,0.5]), 0.02, True)
fluid.connect(boundary.import_particle_systems())




scn.add_node(cube)
scn.add_node(sampler)
scn.add_node(initialParticles)
scn.add_node(fluid)
scn.add_node(boundary)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
#app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
