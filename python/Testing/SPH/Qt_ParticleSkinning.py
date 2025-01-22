from os import supports_fd

import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([1.5, 1, 1.5]))
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))

cube = dyno.CubeModel3f()
scn.add_node(cube)
cube.var_location().set_value(dyno.Vector3f([0.5,0.1,0.5]))
cube.var_length().set_value(dyno.Vector3f([0.04,0.04,0.04]))
cube.set_visible(False)

sampler = dyno.ShapeSampler3f()
scn.add_node(sampler)
sampler.var_sampling_distance().set_value(0.005)
sampler.set_visible(False)

cube.connect(sampler.import_shape())

initialParticles = dyno.MakeParticleSystem3f()
scn.add_node(initialParticles)
sampler.state_point_set().promote_output().connect(initialParticles.in_points())

emitter = dyno.SquareEmitter3f()
scn.add_node(emitter)
emitter.var_location().set_value(dyno.Vector3f([0.5,0.5,0.5]))

fluid = dyno.ParticleFluid3f()
scn.add_node(fluid)
initialParticles.connect(fluid.import_initial_states())
emitter.connect(fluid.import_particle_emitters())

cubeBoundary = dyno.CubeModel3f()
scn.add_node(cubeBoundary)
cubeBoundary.var_location().set_value(dyno.Vector3f([0.5, 0.5,0.5]))
cubeBoundary.var_length().set_value(dyno.Vector3f([1,1,1]))
cubeBoundary.set_visible(False)

cube2vol = dyno.BasicShapeToVolume3f()
scn.add_node(cube2vol)
cube2vol.var_grid_spacing().set_value(0.02)
cube2vol.var_inerted().set_value(True)
cubeBoundary.connect(cube2vol.import_shape())

container = dyno.VolumeBoundary3f()
scn.add_node(container)
cube2vol.connect(container.import_volumes())
fluid.connect(container.import_particle_systems())

meshRe = dyno.ParticleSkinning3f()
scn.add_node(meshRe)
meshRe.state_grid_spacing().set_value(0.005)
fluid.connect(meshRe.import_particle_systems())

marchingCubes = dyno.MarchingCubes3f()
scn.add_node(marchingCubes)
meshRe.state_level_set().connect(marchingCubes.in_level_set())
marchingCubes.var_iso_value().set_value(-300000)
marchingCubes.var_grid_spacing().set_value(0.005)

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.set_color(dyno.Color(0.1,0.1,0.9))
marchingCubes.state_triangle_set().connect(surfaceRenderer.in_triangle_set())
surfaceRenderer.var_alpha().set_value(0.3)
surfaceRenderer.var_metallic().set_value(0.5)
marchingCubes.graphics_pipeline().push_module(surfaceRenderer)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
