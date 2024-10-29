#include <QtApp.h>
#include "initializeModeling.h"

#include "BasicShapes/CubeModel.h"
#include "Commands/Merge.h"
#include "Commands/CopyModel.h"
#include "Commands/Turning.h"
#include "Commands/Extrude.h"
#include "Commands/Sweep.h"

#include "Samplers/PointFromCurve.h"

#include "ObjIO/ObjPointLoader.h"


using namespace dyno;

int main()
{
	//创建场景图
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	//创建Cube
	auto Cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());

	Cube->varLength()->setValue(Vec3f(0.5, 0.2, 0.5));
	Cube->varRotation()->setValue(Vec3f(0, 45, 0));
	Cube->varLocation()->setValue(Vec3f(0, -0.1, 0));

	//创建曲线
	auto Curve = scn->addNode(std::make_shared<ObjPoint<DataType3f>>());
	Curve->varFileName()->setValue(getAssetPath() + "curve/curve06_subdivide.obj");

	//创建Turning模型
	auto Turn = scn->addNode(std::make_shared<TurningModel<DataType3f>>());
	Curve->outPointSet()->connect(Turn->inPointSet());
	//Turn->varColumns()->setValue(50);

	//创建Merge模型
	auto MergeModel = scn->addNode(std::make_shared<Merge<DataType3f>>());
	Cube->stateTriangleSet()->promoteOuput()->connect(MergeModel->inTriangleSet01());
	Turn->stateTriangleSet()->promoteOuput()->connect(MergeModel->inTriangleSet02());


	//创建Copy模型
	auto Copy = scn->addNode(std::make_shared<CopyModel<DataType3f>>());
	MergeModel->stateTriangleSet()->promoteOuput()->connect(Copy->inTriangleSetIn());
	//修改Copy属性
	Copy->varTotalNumber()->setValue(4);
	Copy->varCopyTransform()->setValue(Vec3f(1, 0, 0));

	//Create Curve
	auto curve = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	curve->varRotation()->setValue(dyno::Vec3f(90, 0, 0));
	auto ramp = curve->varCurve()->getValue();

	ramp.useBezier();

	ramp.addPointAndHandlePoint(Ramp::Coord2D(0, 0.5), Ramp::Coord2D(0.25, 0.5), Ramp::Coord2D(0.25, 0.5));
	ramp.addPointAndHandlePoint(Ramp::Coord2D(0.5, 1), Ramp::Coord2D(0.5, 0.75), Ramp::Coord2D(0.5, 0.75));
	ramp.addPointAndHandlePoint(Ramp::Coord2D(1, 0.5), Ramp::Coord2D(0.75, 0.5), Ramp::Coord2D(0.75, 0.5));
	ramp.addPointAndHandlePoint(Ramp::Coord2D(0.5, 0), Ramp::Coord2D(0.5, 0.25), Ramp::Coord2D(0.5, 0.25));

	ramp.setCurveClose(true);
	ramp.setResample(true);
	ramp.setSpacing(5);

	ramp.remapXY(-0.5, 0.5, -0.5, 0.5);
	curve->varCurve()->setValue(ramp);
	//Create Spline
	auto curve2 = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	auto ramp2 = curve2->varCurve()->getValue();

	ramp2.useLinear();

	ramp2.addPoint(0, 0);
	ramp2.addPoint(0, 1);

	ramp2.setCurveClose(false);
	ramp2.setResample(true);
	ramp2.setSpacing(5);
	curve2->varCurve()->setValue(ramp2);

	// Create Sweep
	auto sweep = scn->addNode(std::make_shared<SweepModel<DataType3f>>());

	////Use Obj
	//objcurve->outPointSet()->connect(sweep->inCurve());
	//objspline->outPointSet()->connect(sweep->inSpline());

	curve2->statePointSet()->promoteOuput()->connect(sweep->inSpline());
	curve->statePointSet()->promoteOuput()->connect(sweep->inCurve());

	sweep->varLocation()->setValue(Vec3f(-2, 0, 0));
	{
		auto rampValue = sweep->varCurveRamp()->getData();
		rampValue.addPointAndHandlePoint(Ramp::Coord2D(0, 0.8), Ramp::Coord2D(0.4, 0.8), Ramp::Coord2D(0.4, 0.8));
		rampValue.addPointAndHandlePoint(Ramp::Coord2D(0.5, 0.2), Ramp::Coord2D(0.2, 0.3), Ramp::Coord2D(0.8, 0.3));
		rampValue.addPointAndHandlePoint(Ramp::Coord2D(1, 1), Ramp::Coord2D(0.8, 1), Ramp::Coord2D(0.8, 1));
		sweep->varCurveRamp()->setValue(rampValue);
	}



	//import Curve and Spline
	auto objcurve = scn->addNode(std::make_shared<ObjPoint<DataType3f>>());
	objcurve->varFileName()->setValue(getAssetPath() + "curve/Circle_v15.obj");
	objcurve->varScale()->setValue(Vec3f(0.3, 0.3, 0.3));

	auto objspline = scn->addNode(std::make_shared<ObjPoint<DataType3f>>());
	objspline->varFileName()->setValue(getAssetPath() + "curve/Spline02.obj");
	objspline->varScale()->setValue(Vec3f(0.3, 0.3, 0.3));

	// Create Sweep2
	auto sweepFromOBJ = scn->addNode(std::make_shared<SweepModel<DataType3f>>());
	objcurve->outPointSet()->connect(sweepFromOBJ->inCurve());
	objspline->outPointSet()->connect(sweepFromOBJ->inSpline());

	sweepFromOBJ->varLocation()->setValue(Vec3f(-1, 0, 0));


	Modeling::initStaticPlugin();

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 768);
	app.mainLoop();

	return 0;
}