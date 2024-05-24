import PyPeridyno as dyno

scn = dyno.SceneGraph()
gltf = dyno.GltfLoader3f()
gltf.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Jeep/JeepGltf/jeep.gltf"))

scn.add_node(gltf)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
