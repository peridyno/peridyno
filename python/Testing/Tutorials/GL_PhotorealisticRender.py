from os import supports_fd

import PyPeridyno as dyno

scn = dyno.SceneGraph()

mesh = dyno.TextureMeshLoader3f()
scn.add_node(mesh)
mesh.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "obj/standard/cube.obj"))
mesh.var_scale().set_value(dyno.Vector3f([0.3,0.3,0.3]))
mesh.var_location().set_value(dyno.Vector3f([-1.5,0.3,0]))

mesh1 = dyno.TextureMeshLoader3f()
scn.add_node(mesh1)
mesh1.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "obj/moon/Moon_Normal.obj"))
mesh1.var_scale().set_value(dyno.Vector3f([0.005,0.005,0.005]))
mesh1.var_location().set_value(dyno.Vector3f([0.5,0.3,0.5]))


app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
# app.render_window().get_camera().set_unit_scale(512)
app.main_loop()
