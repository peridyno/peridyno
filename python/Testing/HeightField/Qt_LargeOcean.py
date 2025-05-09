import PyPeridyno as dyno

scn = dyno.SceneGraph()

patch = dyno.OceanPatch3f()
scn.add_node(patch)
patch.var_wind_type().set_value(8)

ocean = dyno.LargeOcean3f()
scn.add_node(ocean)

patch.connect(ocean.import_ocean_patch())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
