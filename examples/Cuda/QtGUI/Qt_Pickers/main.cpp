#include <QtApp.h>
#include <GLRenderEngine.h>
#include "SceneGraph.h"

#include "QuadPickerNode.h"
#include "TrianglePickerNode.h"
#include "EdgePickerNode.h"
#include "PointPickerNode.h"

#include "SurfaceMeshLoader.h"

#include "BasicShapes/CylinderModel.h"
#include "BasicShapes/CubeModel.h"

#include "ParticleSystem/CubeSampler.h"

#include "initializeModeling.h"
#include "initializeInteraction.h"

#include "Mapping/Extract.h"

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

	auto cube1 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube1->varLocation()->setValue(Vec3f(-0.75f, 2.5f, 0.75f));
	cube1->setVisible(false);
	cube1->varSegments()->setValue(Vec3i((uint)6, (uint)6, (uint)6));

	auto quadPickerNode = scn->addNode(std::make_shared<QuadPickerNode<DataType3f>>());
	cube1->stateQuadSet()->connect(quadPickerNode->inTopology());
	//cube1->stateQuadSet()->promoteOuput();

	auto pickerNode = scn->addNode(std::make_shared<TrianglePickerNode<DataType3f>>());
	auto cube2 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube2->varLocation()->setValue(Vec3f(0.75f, 2.5f, -0.75f));
	cube2->setVisible(false);
	cube2->varSegments()->setValue(Vec3i((uint)6, (uint)6, (uint)6));
	cube2->stateTriangleSet()->connect(pickerNode->inTopology());
	//cube2->stateTriangleSet()->promoteOuput();

	auto pointPickerNode = scn->addNode(std::make_shared<PointPickerNode<DataType3f>>());
	auto cube3 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube3->varLocation()->setValue(Vec3f(-0.75f, 0.5f, 0.75f));
	cube3->setVisible(false);
	cube3->varSegments()->setValue(Vec3i((uint)6, (uint)6, (uint)6));
	auto cubeSampler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	cube3->outCube()->connect(cubeSampler->inCube());
	cubeSampler->statePointSet()->connect(pointPickerNode->inTopology());
	//cubeSampler->statePointSet()->promoteOuput();

	auto edgePickerNode = scn->addNode(std::make_shared<EdgePickerNode<DataType3f>>());
	auto cube4 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube4->varLocation()->setValue(Vec3f(0.75f, 0.5f, -0.75f));
	cube4->setVisible(false);
	cube4->varSegments()->setValue(Vec3i((uint)6, (uint)6, (uint)6));
	cube4->stateQuadSet()->connect(edgePickerNode->inTopology());
	//cube4->stateQuadSet()->promoteOuput();

	//auto mesh2 = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
	//mesh2->varFileName()->setValue(getAssetPath() + "submarine/submarine.obj");
	//mesh2->outTriangleSet()->connect(pickerNode->inTopology()); 

//	pickerNode->varInteractionRadius()->setValue(0.02f);
	scn->setUpperBound({ 4, 4, 4 });

	QtApp app;

	app.setSceneGraph(scn);
	app.initialize(1024, 768);

	auto window = app.renderWindow();
	auto cam = window->getCamera();
	cam->setTargetPos(Vec3f(0.0f,1.75f,0.0f));
	cam->setEyePos(Vec3f(3.0f, 3.0f, 3.0f));

	app.mainLoop();

	return 0;
}