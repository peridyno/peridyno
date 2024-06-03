import PyPeridyno as dyno

scene = dyno.SceneGraph()
gltf = dyno.GltfLoader3f()
gltf.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Jeep/JeepGltf/jeep.gltf"))

scene.add_node(gltf)
