import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f(0), pk.Vector3f(1), 0.005, True, False)

for i in range(5):
    root.load_cube(pk.Vector3f([0.2 + i * 0.08, 0.2, 0]), pk.Vector3f([0.25 + i * 0.08, 0.25, 1]), 0.005, False, True)
	
child3 = pk.ParticleViscoplasticBody3f()
root.add_particle_system(child3)

ptRender = pk.PointRenderer()
ptRender.set_color(pk.Vector3f([0, 1, 1]))
child3.add_visual_module(ptRender)

child3.set_mass(1.0)
child3.load_particles("data/bunny_points.obj")
child3.load_surface("data/bunny_mesh.obj")
child3.translate(pk.Vector3f([0.4, 0.4, 0.5]))

child4 = pk.ParticleViscoplasticBody3f()
root.add_particle_system(child4)

ptRender2 = pk.PointRenderer()
ptRender2.set_color(pk.Vector3f([1, 0, 1]))
child4.add_visual_module(ptRender2)

child4.set_mass(1.0)
child4.load_particles("data/bunny_points.obj")
child4.load_surface("data/bunny_mesh.obj")
child4.translate(pk.Vector3f([0.4, 0.4, 0.9]))

# create scene graph
scene = pk.SceneGraph()
scene.set_root_node(root)
