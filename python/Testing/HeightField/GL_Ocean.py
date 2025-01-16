import PyPeridyno as dyno

scn = dyno.SceneGraph()

oceanPatch = dyno.OceanPatch3f()
scn.add_node(oceanPatch)
oceanPatch.var_wind_type().set_value(8)

root = dyno.Ocean3f()
scn.add_node(root)
root.var_extentX().set_value(2)
root.var_extentZ().set_value(2)
oceanPatch.connect(root.import_ocean_patch())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
