import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 1, 1]), 0.005, True, False)

numSegment = 20
numPoint = numSegment + 1
CableStart = pk.Vector3f([15000, 5000, 15000]) * 0.00003 + 0.1
CableEnd = pk.Vector3f([16000, 10000, 15000]) * 0.00003+ 0.1
Cable = CableEnd - CableStart

child3 = pk.ParticleRod3f("Rod")
root.add_particle_system(child3)

child3.set_mass(1.0)
child3.horizon.set_value(Cable.norm() / numSegment)
child3.set_material_stiffness(1.0)

particles = []
for i in range(numPoint):
    pi = CableStart + Cable * i / numSegment
    particles.append(pi)

    if i == 0:
        child3.add_fixed_particle(0, pi)

child3.set_particles(particles)

pointsRender = pk.PointRenderer()
pointsRender.set_point_size(0.003)
child3.add_visual_module(pointsRender)
child3.set_visible(True)

# create scene graph
scene = pk.SceneGraph()	
scene.set_lower_bound(pk.Vector3f([0.0, 0.0, 0.0]))
scene.set_upper_bound(pk.Vector3f([1.0, 1.0, 1.0]))
scene.set_gravity(pk.Vector3f([0.0, -9.8, 0.0]))
scene.set_root_node(root)
