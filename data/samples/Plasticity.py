import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 1, 1]), 0.005, True, False)

child3 = pk.ParticleElastoplasticBody3f()
root.add_particle_system(child3)

ptRender = pk.PointRenderer()
ptRender.set_color(pk.Vector3f([0, 1, 1]))
child3.add_visual_module(ptRender)

child3.set_visible(False)
child3.set_mass(1.0)
child3.load_particles(pk.Vector3f(-1.1), pk.Vector3f(1.15), 0.1)
child3.load_surface("data/standard_cube20.obj")
child3.scale(0.05)
child3.translate(pk.Vector3f([0.3, 0.2, 0.5]))
child3.get_surface_node().set_visible(True)

sRender = pk.SurfaceRenderer()
child3.get_surface_node().add_visual_module(sRender)
sRender.set_color(pk.Vector3f([1, 0, 1]))

child2 = pk.ParticleElasticBody3f()
root.add_particle_system(child2)

ptRender2 = pk.PointRenderer()
ptRender2.set_color(pk.Vector3f([0, 1, 1]))
child2.add_visual_module(ptRender2)

child2.set_visible(False)
child2.set_mass(1.0)
child2.load_particles(pk.Vector3f(-1.1), pk.Vector3f(1.15), 0.1)
child2.load_surface("data/standard_cube20.obj")
child2.scale(0.05)
child2.translate(pk.Vector3f([0.5, 0.2, 0.5]))
child2.get_elasticity_solver().set_iteration_number(10)

sRender2 = pk.SurfaceRenderer()
child2.get_surface_node().add_visual_module(sRender2)
sRender2.set_color(pk.Vector3f([1, 1, 0]))

# create scene graph
scene = pk.SceneGraph()	
scene.set_root_node(root)
