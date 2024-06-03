import PyPhysIKA as pk

# create root node...
root = pk.StaticBoundary3f()
root.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 1, 1]), 0.005, True, False)

# load bunny
body = pk.ParticleElasticBody3f("bunny")
body.load_particles("data/bunny_points.obj")
body.load_surface("data/bunny_mesh.obj")

body.set_mass(1)
body.translate(pk.Vector3f([0.5, 0.5, 0.5]))
root.add_particle_system(body)

# render modules
surfaceColor = pk.Vector3f([0.5, 0.5, 1.])
surfaceMetallic = 0.5
surfaceRoughness = 0.5
surfaceAlpha = 1.

pointColor = pk.Vector3f([1.0, 0.0, 1.])
pointMetallic = 0.
pointRoughness = 1.
pointSize = 0.0015

srender = pk.SurfaceRenderer()
srender.set_color(surfaceColor)
srender.set_metallic(surfaceMetallic)
srender.set_roughness(surfaceRoughness)
srender.set_alpha(surfaceAlpha)
body.get_surface_node().add_visual_module(srender)

prender = pk.PointRenderer()
prender.set_color(pointColor)
prender.set_metallic(pointMetallic)
prender.set_roughness(pointRoughness)
prender.set_point_size(pointSize)
prender.set_colormap_mode(1) # PointRenderer::VELOCITY_JET
prender.set_colormap_range(0, 5)
body.add_visual_module(prender)

# create scene graph
scene = pk.SceneGraph()	
scene.set_frame_rate(60)
scene.set_gravity(pk.Vector3f([0., -9.8, 0.]))
scene.set_root_node(root)
