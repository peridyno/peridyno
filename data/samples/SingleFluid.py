import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([-0.5, 0, -0.5]), pk.Vector3f([1.5, 2, 1.5]), 0.02, True, False)
root.load_sdf("data/bowl.sdf", False)

child1 = pk.ParticleFluid3f()
root.add_particle_system(child1)

ptRender = pk.FluidRenderer()
ptRender.set_color(pk.Vector3f([0, 0, 1]))
ptRender.set_point_size(0.003)
child1.add_visual_module(ptRender)

child1.load_particles(pk.Vector3f([0.5, 0.2, 0.4]), pk.Vector3f([0.7, 1.5, 0.6]), 0.005)
child1.set_mass(100)

rigidbody = pk.RigidBody3f()
root.add_rigid_body(rigidbody)
rigidbody.load_shape("data/bowl.obj")
rigidbody.set_active(False)

sRender2 = pk.SurfaceRenderer()
rigidbody.get_surface().add_visual_module(sRender2)

# create scene graph
scene = pk.SceneGraph()
scene.set_lower_bound(pk.Vector3f([-0.5, 0, -0.5]))
scene.set_upper_bound(pk.Vector3f([1.5, 1, 1.5]))
scene.set_root_node(root)
