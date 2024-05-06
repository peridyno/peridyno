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
scn.set_upper_bound(dyno.Vector3f([1.5, 1, 1.5]))
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))

ptsLoader = dyno.PointsLoader3f()
ptsLoader.var_file_name().set_value(dyno.FilePath(filePath("fish/FishPoints.obj")))
ptsLoader.var_rotation().set_value(dyno.Vector3f([0, 0, 3.1415926]))
ptsLoader.var_location().set_value(dyno.Vector3f([0, 0.15, 0.23]))
initialParticles = dyno.MakeParticleSystem3f()
ptsLoader.out_point_set().promote_output().connect(initialParticles.in_points())

fluid = dyno.ParticleFluid3f()
fluid.var_reshuffle_particles().set_value(True)
initialParticles.connect(fluid.import_initial_states())

fluid.animation_pipeline().clear()

smoothingLength = dyno.FloatingNumber3f()
fluid.animation_pipeline().push_module(smoothingLength)
smoothingLength.var_value().set_value(0.0015)

integrator = dyno.ParticleIntegrator3f()
fluid.state_time_step().connect(integrator.in_time_step())
fluid.state_position().connect(integrator.in_position())
fluid.state_velocity().connect(integrator.in_velocity())
fluid.state_force().connect(integrator.in_force_density())
fluid.animation_pipeline().push_module(integrator)

nbrQuery = dyno.NeighborPointQuery3f()
smoothingLength.out_floating().connect(nbrQuery.in_radius())
fluid.state_position().connect(nbrQuery.in_position())
fluid.animation_pipeline().push_module(nbrQuery)

simple = dyno.SimpleVelocityConstraint3f()


scn.add_node(ptsLoader)
scn.add_node(initialParticles)
scn.add_node(fluid)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
