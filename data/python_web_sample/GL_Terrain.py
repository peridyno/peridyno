import PyPeridyno as dyno

scene = dyno.SceneGraph()

land = dyno.LandScape3f()
land.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "landscape/Landscape_1_Map_1024x1024.png"))
land.var_location().set_value(dyno.Vector3f([0, 100, 0]))
land.var_scale().set_value(dyno.Vector3f([1, 64, 1]))

scene.add_node(land)
