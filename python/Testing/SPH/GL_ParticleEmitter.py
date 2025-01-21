import PyPeridyno as dyno

scn = dyno.SceneGraph()

emitter = dyno.SquareEmitter3f()
scn.add_node(emitter)
emitter.var_location().set_value(dyno.Vector3f([0.5, 0.5, 0.5]))

fluid = dyno.ParticleFluid3f()
scn.add_node(fluid)
emitter.connect(fluid.import_particle_emitters())

cubeBoundary = dyno.CubeModel3f()
scn.add_node(cubeBoundary)
cubeBoundary.var_location().set_value(dyno.Vector3f([0.5,0.5,0.5]))
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

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(800, 600, True)
app.main_loop()
