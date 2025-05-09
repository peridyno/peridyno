from os import supports_fd

import PyPeridyno as dyno

scn = dyno.SceneGraph()

loader = dyno.VolumeLoader3f()
scn.add_node(loader)
loader.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "bowl/bowl.sdf"))

clipper = dyno.VolumeClipper3f()
scn.add_node(clipper)
loader.state_level_set().connect(clipper.in_level_set())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
