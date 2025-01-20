import PyPeridyno as dyno

total_scale = 6

scn = dyno.SceneGraph()

jeep = dyno.Jeep3f()
scn.add_node(jeep)

multibody = dyno.MultibodySystem3f()
scn.add_node(multibody)
jeep.connect(multibody.import_vehicles())
jeep.var_location().set_value(dyno.Vector3f([0, 1, -5]))

ObjLand = dyno.ObjLoader3f()
scn.add_node(ObjLand)
ObjLand.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "landscape/Landscape_resolution_1000_1000.obj"))
ObjLand.var_scale().set_value(dyno.Vector3f([6, 6, 6]))
ObjLand.var_location().set_value(dyno.Vector3f([0, 0, 0.5]))
glLand = ObjLand.graphics_pipeline().find_first_module()
glLand.var_base_color().set_value(dyno.Color(0.82745, 0.82745, 0.82745))
glLand.var_use_vertex_normal().set_value(True)

ObjLand.out_triangle_set().connect(multibody.in_triangle_set())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1366, 768, True)
app.main_loop()
