import os

import PyPeridyno as dyno


def filePath(str):
    script_dir = os.getcwd()
    relative_path = "../../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


scene = dyno.SceneGraph()
scene.set_upper_bound(dyno.Vector3f([1.5, 1, 1.5]))
scene.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))

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
simple.var_viscosity().set_value(500)
simple.var_simple_iteration_enable().set_value(False)
fluid.state_time_step().connect(simple.in_time_step())
smoothingLength.out_floating().connect(simple.in_smoothing_length())
fluid.state_position().connect(simple.in_position())
fluid.state_velocity().connect(simple.in_velocity())

simple.in_sampling_distance().set_value(0.005)
nbrQuery.out_neighbor_ids().connect(simple.in_neighbor_ids())
fluid.animation_pipeline().push_module(simple)

boundary = dyno.StaticBoundary3f()
boundary.load_cube(dyno.Vector3f([-0.5, 0, -0.5]), dyno.Vector3f([1.5, 2, 1.5]), 0.02, True)
boundary.load_sdf(filePath("bowl/bowl.sdf"), False)
fluid.connect(boundary.import_particle_systems())

calculateNorm = dyno.CalculateNorm3f()
fluid.state_velocity().connect(calculateNorm.in_vec())
fluid.graphics_pipeline().push_module(calculateNorm)

colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5)
calculateNorm.out_norm().connect(colorMapper.in_scalar())
fluid.graphics_pipeline().push_module(colorMapper)

ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

fluid.state_point_set().connect(ptRender.in_point_set())
colorMapper.out_color().connect(ptRender.in_color())

fluid.graphics_pipeline().push_module(ptRender)

colorBar = dyno.ImColorbar3f()
colorBar.var_max().set_value(5)
colorBar.var_field_name().set_value("Velocity")
calculateNorm.out_norm().connect(colorBar.in_scalar())
fluid.graphics_pipeline().push_module(colorBar)

scene.add_node(ptsLoader)
scene.add_node(initialParticles)
scene.add_node(fluid)
scene.add_node(boundary)