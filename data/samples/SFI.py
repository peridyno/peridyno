import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 1, 1]), 0.015, True, False)

fluid = pk.ParticleFluid3f()
root.add_particle_system(fluid)

ptRender = pk.PointRenderer()
ptRender.set_color(pk.Vector3f([0, 0, 1]))
ptRender.set_point_size(0.005)
fluid.add_visual_module(ptRender)

fluid.load_particles(pk.Vector3f(0), pk.Vector3f([0.5, 1.0, 1.0]), 0.015)
fluid.set_mass(10)

pbd = pk.PositionBasedFluidModel3f()
fluid.current_position().connect(pbd.position)
fluid.current_velocity().connect(pbd.velocity)
fluid.current_force().connect(pbd.force_density)
pbd.set_smoothing_length(0.02)

fluid.set_numerical_model(pbd)

sfi = pk.SolidFluidInteraction3f()
sfi.set_interaction_distance(0.02)
root.add_child(sfi)

for i in range(3):
    bunny = pk.ParticleElasticBody3f()
    root.add_particle_system(bunny)

    bunny.set_mass(1.0)
    bunny.load_particles("data/sparse_bunny_points.obj")
    bunny.load_surface("data/sparse_bunny_mesh.obj")
    bunny.translate(pk.Vector3f([0.75, 0.2, 0.4 + i * 0.3]))
    bunny.set_visible(False)

    bunny.get_elasticity_solver().set_iteration_number(10)
    bunny.get_elasticity_solver().in_horizon().set_value(0.03)
    bunny.get_topology_mapping().set_searching_radius(0.05)

    sRender = pk.SurfaceRenderer()
    bunny.get_surface_node().add_visual_module(sRender)
    sRender.set_color(pk.Vector3f([i * 0.3, 1 - i * 0.3, 1.0]))
    sRender.set_alpha(0.5)

    sfi.add_particle_system(bunny)

sfi.add_particle_system(fluid)

# create scene graph
scene = pk.SceneGraph()
scene.set_root_node(root)
