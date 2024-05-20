import PyPhysIKA as pk

# create root node...
root = pk.HeightFieldNode3f()
root.load_particles(pk.Vector3f([0, 0.2, 0]), pk.Vector3f([1, 1.5, 1]), 0.005, 0.3, 0.998)
root.set_mass(100)

hRender = pk.HeightFieldRender()
hRender.set_color(pk.Vector3f([1, 0, 0]))
root.add_visual_module(hRender)

# create scene graph
scene = pk.SceneGraph()
scene.set_lower_bound(pk.Vector3f([-0.5, 0, -0.5]))
scene.set_upper_bound(pk.Vector3f([1.5, 1, 1.5]))
scene.set_root_node(root)
