import PyPhysIKA as pk

# create scene graph
scene = pk.SceneGraph()
scene.set_frame_rate(500)
scene.set_upper_bound(pk.Vector3f([1, 2, 1]))
scene.set_lower_bound(pk.Vector3f([0, 0, 0]))

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 2, 1]), 0.015, True, False)

sfi = pk.SolidFluidInteraction3f()
sfi.set_interaction_distance(0.03)

colors = [
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5]
]

for i in range(6):
    bunny = pk.ParticleElasticBody3f("bunny_{}".format(i))

    #bunny.var_horizon().set_value(0.03)
    bunny.set_mass(1.0)
    bunny.load_particles("data/sparse_bunny_points.obj")
    bunny.load_surface("data/sparse_bunny_mesh.obj")
    bunny.translate(pk.Vector3f([0.4, 0.2 + i * 0.3, 0.8]))

    #bunny.getElasticitySolver().setIterationNumber(10)
    #bunny.getTopologyMapping().setSearchingRadius(0.05)

    sfi.add_particle_system(bunny)
    root.add_particle_system(bunny)

    sRender = pk.SurfaceRenderer()
    sRender.set_color(pk.Vector3f(colors[i % 3]))
    bunny.get_surface_node().add_visual_module(sRender)

scene.set_root_node(root)
