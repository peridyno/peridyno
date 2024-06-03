import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 1, 1]), 0.005, True, False)

# first bunny
bunny0 = pk.ParticleElasticBody3f()
root.add_particle_system(bunny0)

bunny0.set_mass(1.0)
bunny0.load_particles("data/bunny_points.obj")
bunny0.load_surface("data/bunny_mesh.obj")
bunny0.translate(pk.Vector3f([0.3, 0.2, 0.5]))

hyperElasicity = pk.HyperelasticityModule3f()
hyperElasicity.set_energy_function(pk.HyperelasticityModule3f.EnergyType.Quadratic)
hyperElasicity.set_iteration_number(10)
bunny0.set_elasticity_solver(hyperElasicity)

sRender0 = pk.SurfaceRenderer()
bunny0.get_surface_node().add_visual_module(sRender0)
sRender0.set_color(pk.Vector3f([1, 1, 0]))
sRender0.set_alpha(0.5)

# second bunny
bunny1 = pk.ParticleElasticBody3f()
root.add_particle_system(bunny1)

bunny1.set_mass(1.0)
bunny1.load_particles("data/bunny_points.obj")
bunny1.load_surface("data/bunny_mesh.obj")
bunny1.translate(pk.Vector3f([0.7, 0.2, 0.5]))

sRender1 = pk.SurfaceRenderer()
sRender1.set_color(pk.Vector3f([0, 1, 1]))
sRender1.set_alpha(0.5)
bunny1.get_surface_node().add_visual_module(sRender1)

# create scene graph
scene = pk.SceneGraph()	
scene.set_root_node(root)
