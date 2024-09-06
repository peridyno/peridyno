#include <QtApp.h>

#include "initializeModeling.h"
#include "Commands/PointFromCurve.h"
#include "Commands/Sweep.h"
#include "Commands/Extrude.h"

#include "ObjIO/OBJexporter.h"

#include "Curve.h"
#include "Ramp.h"
#include "Canvas.h"



using namespace dyno;

int main()
{
	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	////Curve 
	////Create Curve
	//auto curve = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	//curve->varRotation()->setValue(dyno::Vec3f(90, 0, 0));
	//auto ramp = curve->varCurve()->getValue();

	//ramp.useBezier();

	//ramp.addPointAndHandlePoint(Ramp::Coord2D(0, 0.5), Ramp::Coord2D(0.25, 0.5), Ramp::Coord2D(0.25, 0.5));
	//ramp.addPointAndHandlePoint(Ramp::Coord2D(0.5, 1), Ramp::Coord2D(0.5, 0.75), Ramp::Coord2D(0.5, 0.75));
	//ramp.addPointAndHandlePoint(Ramp::Coord2D(1, 0.5), Ramp::Coord2D(0.75, 0.5), Ramp::Coord2D(0.75, 0.5));
	//ramp.addPointAndHandlePoint(Ramp::Coord2D(0.5, 0), Ramp::Coord2D(0.5, 0.25), Ramp::Coord2D(0.5, 0.25));

	//ramp.setCurveClose(true);
	//ramp.setResample(true);
	//ramp.setSpacing(5);

	//ramp.remapXY(-0.5,0.5,-0.5,0.5);
	//curve->varCurve()->setValue(ramp);



	////Create Spline
	//auto curve2 = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	//auto ramp2 = curve2->varCurve()->getValue();

	//ramp2.useLinear();

	//ramp2.addPoint(0, 0);
	//ramp2.addPoint(0, 1);

	//ramp2.setCurveClose(false);
	//ramp2.setResample(true);
	//ramp2.setSpacing(5);
	//curve2->varCurve()->setValue(ramp2);

	////Create Line
	//auto sweep = scn->addNode(std::make_shared<SweepModel<DataType3f>>());


	//auto sweepRamp = sweep->varCurveRamp()->getData();
	//sweepRamp.useBezier();

	//sweepRamp.addPointAndHandlePoint(Ramp::Coord2D(0, 0.8), Ramp::Coord2D(0.4, 0.8), Ramp::Coord2D(0.4, 0.8));
	//sweepRamp.addPointAndHandlePoint(Ramp::Coord2D(0.5, 0.2), Ramp::Coord2D(0.2, 0.3), Ramp::Coord2D(0.8, 0.3));
	//sweepRamp.addPointAndHandlePoint(Ramp::Coord2D(1, 1), Ramp::Coord2D(0.8, 1), Ramp::Coord2D(0.8, 1));

	//sweep->varCurveRamp()->setValue(sweepRamp);
	//sweep->varRadius()->setValue(0.5);
	//sweep->varLocation()->setValue(Vec3f(-1.1,0,0));
	//sweep->varScale()->setValue(Vec3f(1,0.5,1));
	//curve->statePointSet()->connect(sweep->inCurve());
	//curve2->statePointSet()->connect(sweep->inSpline());
	//sweep->varDisplayPoints()->setValue(true);
	//sweep->varDisplayWireframe()->setValue(true);


	////Create Curve E1
	//auto curveE1 = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	//curveE1->varRotation()->setValue(dyno::Vec3f(0, 0, 0));
	//curveE1->varLocation()->setValue(Vec3f(0.9,0,0));

	//auto rampE1 = std::make_shared<Curve>(curveE1->varCurve()->getValue());

	//rampE1->useBezier();

	//rampE1->addPointAndHandlePoint(Canvas::Coord2D(0, 0.5), Canvas::Coord2D(0.25, 0.5), Canvas::Coord2D(0.25, 0.5));
	//rampE1->addPointAndHandlePoint(Canvas::Coord2D(0.5, 1), Canvas::Coord2D(0.5, 0.75), Canvas::Coord2D(0.5, 0.75));
	//rampE1->addPointAndHandlePoint(Canvas::Coord2D(1, 0.5), Canvas::Coord2D(0.75, 0.5), Canvas::Coord2D(0.75, 0.5));
	//rampE1->addPointAndHandlePoint(Canvas::Coord2D(0.5, 0), Canvas::Coord2D(0.5, 0.25), Canvas::Coord2D(0.5, 0.25));

	//rampE1->setCurveClose(true);
	//rampE1->setResample(true);
	//rampE1->setSpacing(5);

	//curveE1->varCurve()->setValue(*rampE1);

	////Create Curve E2
	//auto curveE2 = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	//curveE2->varRotation()->setValue(dyno::Vec3f(0, 0, 0));
	//curveE2->varLocation()->setValue(Vec3f(2, 0, 0));
	//auto rampE2 = std::make_shared<Curve>(curveE2->varCurve()->getValue());

	//rampE2->useBezier();

	//rampE2->addPointAndHandlePoint(Canvas::Coord2D(0, 0.5), Canvas::Coord2D(0, 1), Canvas::Coord2D(0, 0));
	//rampE2->addPointAndHandlePoint(Canvas::Coord2D(0.5, 0), Canvas::Coord2D(0, 0), Canvas::Coord2D(1, 0));
	//rampE2->addPointAndHandlePoint(Canvas::Coord2D(1, 0.5), Canvas::Coord2D(1, 0), Canvas::Coord2D(1, 1));
	//rampE2->addPointAndHandlePoint(Canvas::Coord2D(0.5, 1), Canvas::Coord2D(1, 1), Canvas::Coord2D(0, 1));

	//rampE2->setCurveClose(true);
	//rampE2->setResample(true);
	//rampE2->setSpacing(5);
	//
	//curveE2->varCurve()->setValue(*rampE2);


	////Create Curve E3
	//auto curveE3 = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	//curveE3->varRotation()->setValue(dyno::Vec3f(0, 0, 0));
	//curveE3->varLocation()->setValue(Vec3f(3.1, 0, 0));

	//auto rampE3 = std::make_shared<Curve>(curveE3->varCurve()->getValue());

	//rampE3->useLinear();

	//rampE3->addPointAndHandlePoint(Canvas::Coord2D(0, 0.5), Canvas::Coord2D(0, 1), Canvas::Coord2D(0, 0));
	//rampE3->addPointAndHandlePoint(Canvas::Coord2D(0.5, 0), Canvas::Coord2D(0, 0), Canvas::Coord2D(1, 0));
	//rampE3->addPointAndHandlePoint(Canvas::Coord2D(1, 0.5), Canvas::Coord2D(1, 0), Canvas::Coord2D(1, 1));
	//rampE3->addPointAndHandlePoint(Canvas::Coord2D(0.5, 1), Canvas::Coord2D(1, 1), Canvas::Coord2D(0, 1));

	//rampE3->setCurveClose(true);
	//rampE3->setResample(true);
	//rampE3->setSpacing(5);
	//curveE3->varCurve()->setValue(*rampE3);

	////Create Curve E4
	//auto curveE4 = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	//curveE4->varRotation()->setValue(dyno::Vec3f(0, 0, 0));
	//curveE4->varLocation()->setValue(Vec3f(4.2, 0, 0));
	//auto rampE4 = std::make_shared<Curve>(curveE4->varCurve()->getValue());

	//rampE4->useLinear();

	//rampE4->addPointAndHandlePoint(Canvas::Coord2D(0, 0.5), Canvas::Coord2D(0, 1), Canvas::Coord2D(0, 0));
	//rampE4->addPointAndHandlePoint(Canvas::Coord2D(0.5, 0), Canvas::Coord2D(0, 0), Canvas::Coord2D(1, 0));
	//rampE4->addPointAndHandlePoint(Canvas::Coord2D(1, 0.5), Canvas::Coord2D(1, 0), Canvas::Coord2D(1, 1));
	//rampE4->addPointAndHandlePoint(Canvas::Coord2D(0.5, 1), Canvas::Coord2D(1, 1), Canvas::Coord2D(0, 1));

	//rampE4->setCurveClose(false);
	//rampE4->setResample(true);
	//rampE4->setSpacing(5);

	//curveE4->varCurve()->setValue(*rampE4);

	////Create Curve E5
	//auto curveE5 = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	//curveE5->varRotation()->setValue(Vec3f(0, 0, 0));
	//curveE5->varLocation()->setValue(Vec3f(5.3, 0, 0));
	//auto rampE5 = std::make_shared<Curve>(curveE5->varCurve()->getValue());

	//rampE5->useLinear();

	//rampE5->addPointAndHandlePoint(Canvas::Coord2D(0, 0.5), Canvas::Coord2D(0, 1), Canvas::Coord2D(0, 0));
	//rampE5->addPointAndHandlePoint(Canvas::Coord2D(0.5, 0), Canvas::Coord2D(0, 0), Canvas::Coord2D(1, 0));
	//rampE5->addPointAndHandlePoint(Canvas::Coord2D(1, 0.5), Canvas::Coord2D(1, 0), Canvas::Coord2D(1, 1));
	//rampE5->addPointAndHandlePoint(Canvas::Coord2D(0.5, 1), Canvas::Coord2D(1, 1), Canvas::Coord2D(0, 1));

	//rampE5->setCurveClose(false);
	//rampE5->setResample(true);
	//rampE5->setSpacing(2);

	//curveE5->varCurve()->setValue(*rampE5);


	//Create Curve E6
	auto curveE6 = scn->addNode(std::make_shared<PointFromCurve<DataType3f>>());
	curveE6->varRotation()->setValue(Vec3f(90, 0, 0));
	curveE6->varLocation()->setValue(Vec3f(-2.7, 0, -0.5));
	auto rampE6 = std::make_shared<Curve>(curveE6->varCurve()->getValue());

	rampE6->useLinear();

	rampE6->addPointAndHandlePoint(Canvas::Coord2D(0.1, 0.6), Canvas::Coord2D(0.1, 0.6), Canvas::Coord2D(0.1, 0.6));
	rampE6->addPointAndHandlePoint(Canvas::Coord2D(0.1, 0.4), Canvas::Coord2D(0.1, 0.4), Canvas::Coord2D(0.1, 0.4));
	rampE6->addPointAndHandlePoint(Canvas::Coord2D(0.5, 0.3), Canvas::Coord2D(0.5, 0.3), Canvas::Coord2D(0.5, 0.3));
	rampE6->addPointAndHandlePoint(Canvas::Coord2D(0.5, 0.1), Canvas::Coord2D(0.5, 0.1), Canvas::Coord2D(0.5, 0.1));
	rampE6->addPointAndHandlePoint(Canvas::Coord2D(0.85, 0.5), Canvas::Coord2D(0.85, 0.5), Canvas::Coord2D(0.85, 0.5));
	rampE6->addPointAndHandlePoint(Canvas::Coord2D(0.5, 0.8), Canvas::Coord2D(0.5, 0.8), Canvas::Coord2D(0.5, 0.8));
	rampE6->addPointAndHandlePoint(Canvas::Coord2D(0.35, 0.65), Canvas::Coord2D(0.35, 0.65), Canvas::Coord2D(0.35, 0.65));


	rampE6->setCurveClose(true);
	rampE6->setResample(false);
	rampE6->setSpacing(2);

	curveE6->varCurve()->setValue(*rampE6);

	//ExtrudeModel 
	auto extrude = scn->addNode(std::make_shared<ExtrudeModel<DataType3f>>());
	curveE6->statePointSet()->connect(extrude->inPointSet());
	extrude->varLocation()->setValue(Vec3f(0,0,0));
	extrude->varHeight()->setValue(0.5);
	
	






	Modeling::initStaticPlugin();


	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}