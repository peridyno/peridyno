import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()

root.load_sdf("data/bar.sdf", False)
root.translate(pk.Vector3f([0.2, 0.2, 0]))
root.load_cube(pk.Vector3f(0), pk.Vector3f(1), 0.005, True, False)

child3 = pk.ParticleElastoplasticBody3f()
root.add_particle_system(child3)

ptRender = pk.PointRenderer()
ptRender.set_color(pk.Vector3f([0.50, 0.44, 0.38]))
ptRender.set_point_size(0.003)
child3.add_visual_module(ptRender)

child3.set_mass(1.0)
child3.load_particles("data/bunny_points.obj")
child3.load_surface("data/bunny_mesh.obj")
child3.translate(pk.Vector3f([0.3, 0.4, 0.5]))
child3.set_Dt(0.001)

elasto = pk.GranularModule3f()
elasto.enable_fully_reconstruction()
elasto.set_cohesion(0.001)
child3.set_elastoplasticity_solver(elasto)

rigidbody = pk.RigidBody3f()
root.add_rigid_body(rigidbody)
rigidbody.load_shape("data/bar.obj")
rigidbody.set_active(False)
rigidbody.translate(pk.Vector3f([0.2, 0.2, 0]))

sRender = pk.SurfaceRenderer()
rigidbody.get_surface().add_visual_module(sRender)

# create scene graph
scene = pk.SceneGraph()
scene.set_root_node(root)
