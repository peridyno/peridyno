import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 1, 1]), 0.005, True, False)

child1 = pk.ParticleFluid3f()
root.add_particle_system(child1)

ptRender1 = pk.PointRenderer()
ptRender1.set_color(pk.Vector3f([1, 0, 0]))
ptRender1.set_point_size(0.005)
child1.add_visual_module(ptRender1)

child1.load_particles("data/fluid_point.obj")
child1.set_mass(100)
child1.scale(2)
child1.translate(pk.Vector3f([-0.6, -0.3, -0.48]))

multifluid = pk.MultipleFluidModel3f()
child1.current_position().connect(multifluid.position)
child1.current_velocity().connect(multifluid.velocity)
child1.current_force().connect(multifluid.force_density)
#multifluid.m_color.connect(&ptRender1.m_vecIndex)

child1.set_numerical_model(multifluid)

# create scene graph
scene = pk.SceneGraph()	
scene.set_root_node(root)
