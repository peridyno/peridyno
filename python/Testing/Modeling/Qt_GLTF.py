import PyPeridyno as dyno

scn = dyno.SceneGraph()

gltf = dyno.GltfLoader3f()
scn.add_node(gltf)

gltf.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Jeep/JeepGltf/jeep.gltf"))


app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
