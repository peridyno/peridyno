import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 0.5, 0.5]), 0.005, True, False)
root.load_cube(pk.Vector3f([-0.1, 0, -0.1]), pk.Vector3f([0.1, 0.25, 0.6]), 0.005, False, True)

# body
body = pk.ParticleElastoplasticBody3f()
root.add_particle_system(body)
body.set_mass(1)
body.load_particles(pk.Vector3f([0, 0.25, 0.1]), pk.Vector3f([0.2, 0.4, 0.4]), 0.005)

# solver
fracture = pk.FractureModule3f()
body.set_elastoplasticity_solver(fracture)

# visual module
prender = pk.PointRenderer()
prender.set_color(pk.Vector3f([0, 1, 1]))
body.add_visual_module(prender)

# create scene graph
scene = pk.SceneGraph()	
scene.set_lower_bound(pk.Vector3f([0, 0, 0]))
scene.set_upper_bound(pk.Vector3f([1, 0.5, 0.5]))
scene.set_root_node(root)
