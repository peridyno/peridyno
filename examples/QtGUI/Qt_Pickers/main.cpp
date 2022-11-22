#include <QtApp.h>
#include <GLRenderEngine.h>
#include "SceneGraph.h"

#include "PickerNode.h"

#include "SurfaceMeshLoader.h"

#include "CylinderModel.h"
#include "CubeModel.h"

using namespace dyno;

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto pickerNode = scn->addNode(std::make_shared<PickerNode<DataType3f>>());

	auto mesh1 = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
	mesh1->varFileName()->setValue(getAssetPath() + "standard/standard_sphere.obj");
	//mesh1->outTriangleSet()->connect(pickerNode->inTopology()); 

	auto cylinder = scn->addNode(std::make_shared<CylinderModel<DataType3f>>());
	cylinder->setVisible(false);
	cylinder->stateTriangleSet()->connect(pickerNode->inTopology());

	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->setVisible(false);
	cube->varSegments()->setValue(Vec3i((uint)4, (uint)4, (uint)4));
	//cube->stateQuadSet()->connect(pickerNode->inTopology());

	auto mesh2 = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
	mesh2->varFileName()->setValue(getAssetPath() + "submarine/submarine.obj");
	//mesh2->outTriangleSet()->connect(pickerNode->inTopology()); 

	pickerNode->varInterationRadius()->setValue(0.02f);
	scn->setUpperBound({ 4, 4, 4 });

	QtApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}