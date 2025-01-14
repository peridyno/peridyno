#include <UbiApp.h>
#include <SceneGraph.h>

#include <BasicShapes/SphereModel.h>

#include "Volume/VolumeGenerator.h"
#include <Volume/VolumeBoolean.h>
#include <Volume/VolumeClipper.h>
#include <Volume/MarchingCubes.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene1()
{
	//For uniform SDF boolean
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(2, 2, 2));
	scn->setLowerBound(Vec3f(-2, -2, -2));

	auto mesh1 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	mesh1->varLocation()->setValue(Vec3f(-0.2f, 0.0f, 0.0f));
	mesh1->varType()->setCurrentKey(SphereModel<DataType3f>::Icosahedron);
	mesh1->varIcosahedronStep()->setValue(3);

	auto sdfUniformA = scn->addNode(std::make_shared<VolumeGenerator<DataType3f>>());
	sdfUniformA->varSpacing()->setValue(0.069f);

	mesh1->stateTriangleSet()->connect(sdfUniformA->inTriangleSet());


	auto mesh2 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	mesh2->varLocation()->setValue(Vec3f(0.2f, 0.0f, 0.0f));
	mesh2->varType()->setCurrentKey(SphereModel<DataType3f>::Icosahedron);
	mesh2->varIcosahedronStep()->setValue(3);

	auto sdfUniformB = scn->addNode(std::make_shared<VolumeGenerator<DataType3f>>());
	sdfUniformB->varSpacing()->setValue(0.05f);

	mesh2->stateTriangleSet()->connect(sdfUniformB->inTriangleSet());


	auto volumeBool1 = scn->addNode(std::make_shared<VolumeBoolean<DataType3f>>());
	volumeBool1->varSpacing()->setValue(0.05f);
	sdfUniformA->outLevelSet()->connect(volumeBool1->inA());
	sdfUniformB->outLevelSet()->connect(volumeBool1->inB());

	auto clipper = scn->addNode(std::make_shared<VolumeClipper<DataType3f>>());
	volumeBool1->outLevelSet()->connect(clipper->inLevelSet());

// 	auto marchingCubes = scn->addNode(std::make_shared<MarchingCubes<DataType3f>>());
// 	volumeBool1->outLevelSet()->connect(marchingCubes->inLevelSet());
// 	marchingCubes->varIsoValue()->setValue(-0.0f);

	return scn;
}

int main()
{
	UbiApp window(GUIType::GUI_QT);

	window.setSceneGraph(createScene1());
	// window.createWindow(2048, 1152);
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}


