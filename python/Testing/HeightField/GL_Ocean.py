
import sys
print(sys.path)
import PyPeridyno as dyno

scn = dyno.SceneGraph()

oceanPatch = dyno.OceanPatch3f()
oceanPatch.var_wind_type().set_value(8)

root = dyno.Ocean3f()
root.var_extentX().set_value(2)
root.var_extentz().set_value(2)
oceanPatch.connect(root.import_ocean_patch())

waves = dyno.CapillaryWave3f()
waves.connnect(root.import_capillary_waves())

mapper = dyno.HeightFieldToTriangleSet3f()

scn.add_node(oceanPatch)
scn.add_node(root)
scn.add_node(waves)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(800, 600, True)
app.main_loop()