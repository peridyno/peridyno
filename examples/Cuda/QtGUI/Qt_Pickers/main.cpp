#include <QtApp.h>
#include <GLRenderEngine.h>
#include "SceneGraph.h"

#include "PickerNode.h"
#include "EdgePickerNode.h"
#include "PointPickerNode.h"

#include "SurfaceMeshLoader.h"

#include "CylinderModel.h"
#include "CubeModel.h"
#include "CubeSampler.h"

#include "initializeModeling.h"
#include "initializeInteraction.h"

using namespace dyno;

int main()
{
	Modeling::initStaticPlugin();
	Interaction::initStaticPlugin();

	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	/*auto pickerNode = scn->addNode(std::make_shared<PickerNode<DataType3f>>());
	auto mesh1 = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
	mesh1->varFileName()->setValue(getAssetPath() + "standard/standard_sphere.obj");
	mesh1->outTriangleSet()->connect(pickerNode->inTopology()); */

	//auto cylinder = scn->addNode(std::make_shared<CylinderModel<DataType3f>>());
	//cylinder->setVisible(false);
	//cylinder->stateTriangleSet()->connect(pickerNode->inTopology());

	auto pickerNode = scn->addNode(std::make_shared<PickerNode<DataType3f>>());
	auto cube1 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube1->varLocation()->setValue(Vec3f(0.0f, 0.0f, 0.0f));
	cube1->setVisible(false);
	cube1->varSegments()->setValue(Vec3i((uint)6, (uint)6, (uint)6));
	cube1->stateQuadSet()->connect(pickerNode->inTopology());
	cube1->stateQuadSet()->promoteOuput();

	auto pointPickerNode = scn->addNode(std::make_shared<PointPickerNode<DataType3f>>());
	auto cube2 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube2->varLocation()->setValue(Vec3f(-1.25f, 0.0f, 1.25f));
	cube2->setVisible(false);
	cube2->varSegments()->setValue(Vec3i((uint)6, (uint)6, (uint)6));
	auto cubeSampler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	cube2->outCube()->connect(cubeSampler->inCube());
	cubeSampler->statePointSet()->connect(pointPickerNode->inTopology());
	cubeSampler->statePointSet()->promoteOuput();

	auto edgePickerNode = scn->addNode(std::make_shared<EdgePickerNode<DataType3f>>());
	auto cube3 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube3->varLocation()->setValue(Vec3f(1.25f, 0.0f, -1.25f));
	cube3->setVisible(false);
	cube3->varSegments()->setValue(Vec3i((uint)6, (uint)6, (uint)6));
	cube3->stateQuadSet()->connect(edgePickerNode->inTopology());
	cube3->stateQuadSet()->promoteOuput();

	//auto mesh2 = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
	//mesh2->varFileName()->setValue(getAssetPath() + "submarine/submarine.obj");
	//mesh2->outTriangleSet()->connect(pickerNode->inTopology()); 

	pickerNode->varInteractionRadius()->setValue(0.02f);
	scn->setUpperBound({ 4, 4, 4 });

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}