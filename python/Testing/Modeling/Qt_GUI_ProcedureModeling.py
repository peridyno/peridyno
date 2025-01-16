import PyPeridyno as dyno

scn = dyno.SceneGraph()

Cube = dyno.CubeModel3f()
scn.add_node(Cube)

Cube.var_length().set_value(dyno.Vector3f([0.5, 0.2, 0.5]))
Cube.var_rotation().set_value(dyno.Vector3f([0, 45, 0]))
Cube.var_location().set_value(dyno.Vector3f([0, -0.1, 0]))

Curve = dyno.ObjPoint3f()
scn.add_node(Curve)
Curve.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "curve/curve06_subdivide.obj"))

Turn = dyno.TurningModel3f()
scn.add_node(Turn)
Curve.out_point_set().connect(Turn.in_point_set())

MergeModel = dyno.Merge3f()
scn.add_node(MergeModel)
Cube.state_triangle_set().promote_output().connect(MergeModel.in_triangle_set_01())
Turn.state_triangle_set().promote_output().connect(MergeModel.in_triangle_set_02())

Copy = dyno.CopyModel3f()
scn.add_node(Copy)
MergeModel.state_triangle_set().promote_output().connect(Copy.in_triangle_set_in())

Copy.var_total_number().set_value(4)
Copy.var_copy_transform().set_value(dyno.Vector3f([1,0,0]))

curve = dyno.PointFromCurve3f()
scn.add_node(curve)
curve.var_rotation().set_value(dyno.Vector3f([90, 0, 0]))

ramp = curve.var_curve().get_value()

ramp.use_bezier()

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
