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

ramp.add_point_and_handle_point(dyno.Ramp.Coord2D(0, 0.5), dyno.Ramp.Coord2D(0.25, 0.5), dyno.Ramp.Coord2D(0.25, 0.5))
ramp.add_point_and_handle_point(dyno.Ramp.Coord2D(0.5, 1), dyno.Ramp.Coord2D(0.5, 0.75), dyno.Ramp.Coord2D(0.5, 0.75))
ramp.add_point_and_handle_point(dyno.Ramp.Coord2D(1, 0.5), dyno.Ramp.Coord2D(0.75, 0.5), dyno.Ramp.Coord2D(0.75, 0.5))
ramp.add_point_and_handle_point(dyno.Ramp.Coord2D(0.5, 0), dyno.Ramp.Coord2D(0.5, 0.25), dyno.Ramp.Coord2D(0.5, 0.25))

ramp.set_curve_close(True)
ramp.set_resample(True)
ramp.set_spacing(5)

ramp.remap_xy(-0.5, 0.5, -0.5, 0.5)
curve.var_curve().set_value(ramp)

curve2 = dyno.PointFromCurve3f()
scn.add_node(curve2)
ramp2 = curve2.var_curve().get_value()

ramp2.use_linear()

ramp2.add_point(0, 0)
ramp2.add_point(0, 1)

ramp2.set_curve_close(False)
ramp2.set_resample(True)
ramp2.set_spacing(5)
curve2.var_curve().set_value(ramp2)

sweep = dyno.SweepModel3f()
scn.add_node(sweep)

curve2.state_point_set().promote_output().connect(sweep.in_spline())
curve.state_point_set().promote_output().connect(sweep.in_curve())

sweep.var_location().set_value(dyno.Vector3f([-2, 0, 0]))

rampValue = sweep.var_curve_ramp().get_data()
rampValue.add_point_and_handle_point(dyno.Ramp.Coord2D(0, 0.8), dyno.Ramp.Coord2D(0.4, 0.8), dyno.Ramp.Coord2D(0.4, 0.8))
rampValue.add_point_and_handle_point(dyno.Ramp.Coord2D(0.5, 0.2), dyno.Ramp.Coord2D(0.2, 0.3), dyno.Ramp.Coord2D(0.8, 0.3))
rampValue.add_point_and_handle_point(dyno.Ramp.Coord2D(1, 1), dyno.Ramp.Coord2D(0.8, 1), dyno.Ramp.Coord2D(0.8, 1))
sweep.var_curve_ramp().set_value(rampValue)

# import Curve and Spline
objcurve = dyno.ObjPoint3f()
scn.add_node(objcurve)
objcurve.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "curve/Circle_v15.obj"))
objcurve.var_scale().set_value(dyno.Vector3f([0.3, 0.3, 0.3]))

objspline = dyno.ObjPoint3f()
scn.add_node(objspline)
objspline.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "curve/Spline02.obj"))
objspline.var_scale().set_value(dyno.Vector3f([0.3, 0.3, 0.3]))

# Create Sweep2
sweepFromOBJ = dyno.SweepModel3f()
scn.add_node(sweepFromOBJ)
objcurve.out_point_set().connect(sweepFromOBJ.in_curve())
objspline.out_point_set().connect(sweepFromOBJ.in_spline())

sweepFromOBJ.var_location().set_value(dyno.Vector3f([-1, 0, 0]))

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
