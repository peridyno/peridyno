import PyPeridyno as dyno

scn = dyno.SceneGraph()

Cube = dyno.CubeModel3f()
scn.addNode(Cube)

Cube.varLength().setValue(dyno.Vector3f([0.5, 0.2, 0.5]))
Cube.varRotation().setValue(dyno.Vector3f([0, 45, 0]))
Cube.varLocation().setValue(dyno.Vector3f([0, -0.1, 0]))

Curve = dyno.ObjPoint3f()
scn.addNode(Curve)
Curve.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "curve/curve06_subdivide.obj"))

Turn = dyno.TurningModel3f()
scn.addNode(Turn)
Curve.outPointSet().connect(Turn.inPointSet())

MergeModel = dyno.Merge3f()
scn.addNode(MergeModel)
Cube.stateTriangleSet().promoteOuput().connect(MergeModel.inTriangleSets())
Turn.stateTriangleSet().promoteOuput().connect(MergeModel.inTriangleSets())

Copy = dyno.CopyModel3f()
scn.addNode(Copy)
MergeModel.stateTriangleSet().promoteOuput().connect(Copy.inTriangleSetIn())

Copy.varTotalNumber().setValue(4)
Copy.varCopyTransform().setValue(dyno.Vector3f([1,0,0]))

curve = dyno.PointFromCurve3f()
scn.addNode(curve)
curve.varRotation().setValue(dyno.Vector3f([90, 0, 0]))

ramp = curve.varCurve().getValue()

ramp.useBezier()

ramp.addPointAndHandlePoint(dyno.CanvasCoord2D(0, 0.5), dyno.CanvasCoord2D(0.25, 0.5), dyno.CanvasCoord2D(0.25, 0.5))
ramp.addPointAndHandlePoint(dyno.CanvasCoord2D(0.5, 1), dyno.CanvasCoord2D(0.5, 0.75), dyno.CanvasCoord2D(0.5, 0.75))
ramp.addPointAndHandlePoint(dyno.CanvasCoord2D(1, 0.5), dyno.CanvasCoord2D(0.75, 0.5), dyno.CanvasCoord2D(0.75, 0.5))
ramp.addPointAndHandlePoint(dyno.CanvasCoord2D(0.5, 0), dyno.CanvasCoord2D(0.5, 0.25), dyno.CanvasCoord2D(0.5, 0.25))

ramp.setCurveClose(True)
ramp.setResample(True)
ramp.setSpacing(5)

curve.varCurve().setValue(ramp)

curve2 = dyno.PointFromCurve3f()
scn.addNode(curve2)
ramp2 = curve2.varCurve().getValue()

ramp2.useLinear()

ramp2.addPoint(0, 0)
ramp2.addPoint(0, 1)

ramp2.setCurveClose(False)
ramp2.setResample(True)
ramp2.setSpacing(5)
curve2.varCurve().setValue(ramp2)

sweep = dyno.SweepModel3f()
scn.addNode(sweep)

curve2.statePointSet().promoteOuput().connect(sweep.inSpline())
curve.statePointSet().promoteOuput().connect(sweep.inCurve())

sweep.varLocation().setValue(dyno.Vector3f([-2, 0, 0]))

rampValue = sweep.varCurveRamp().getData()
rampValue.addPointAndHandlePoint(dyno.CanvasCoord2D(0, 0.8), dyno.CanvasCoord2D(0.4, 0.8), dyno.CanvasCoord2D(0.4, 0.8))
rampValue.addPointAndHandlePoint(dyno.CanvasCoord2D(0.5, 0.2), dyno.CanvasCoord2D(0.2, 0.3), dyno.CanvasCoord2D(0.8, 0.3))
rampValue.addPointAndHandlePoint(dyno.CanvasCoord2D(1, 1), dyno.CanvasCoord2D(0.8, 1), dyno.CanvasCoord2D(0.8, 1))
sweep.varCurveRamp().setValue(rampValue)

# import Curve and Spline
objcurve = dyno.ObjPoint3f()
scn.addNode(objcurve)
objcurve.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "curve/Circle_v15.obj"))
objcurve.varScale().setValue(dyno.Vector3f([0.3, 0.3, 0.3]))

objspline = dyno.ObjPoint3f()
scn.addNode(objspline)
objspline.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "curve/Spline02.obj"))
objspline.varScale().setValue(dyno.Vector3f([0.3, 0.3, 0.3]))

# Create Sweep2
sweepFromOBJ = dyno.SweepModel3f()
scn.addNode(sweepFromOBJ)
objcurve.outPointSet().connect(sweepFromOBJ.inCurve())
objspline.outPointSet().connect(sweepFromOBJ.inSpline())

sweepFromOBJ.varLocation().setValue(dyno.Vector3f([-1, 0, 0]))

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
