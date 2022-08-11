#include <QtApp.h>
#include <GLRenderEngine.h>
#include "SceneGraph.h"

#include "Node/GLSurfaceVisualNode.h"
#include "Node/GLCommonPointVisualNode.h"
#include "Node/GLWireframeVisualNode.h"

//#include "SurfacePickerNode.h"
//#include "PointPickerNode.h"
//#include "EdgePickerNode.h"
#include "PickerNode.h"

#include "SurfaceMeshLoader.h"

using namespace dyno;

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto mesh = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
	/*auto surfacePickerNode = scn->addNode(std::make_shared<SurfacePickerNode<DataType3f>>());
	auto edgePickerNode = scn->addNode(std::make_shared<EdgePickerNode<DataType3f>>());
	auto pointPickerNode = scn->addNode(std::make_shared<PointPickerNode<DataType3f>>());*/
	auto pickerNode = scn->addNode(std::make_shared<PickerNode<DataType3f>>());

	//auto sRender1 = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	//auto sRender2 = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	//auto wRender1 = scn->addNode(std::make_shared<GLWireframeVisualNode<DataType3f>>());
	//auto wRender2 = scn->addNode(std::make_shared<GLWireframeVisualNode<DataType3f>>());
	//auto pRender1 = scn->addNode(std::make_shared<GLCommonPointVisualNode<DataType3f>>());
	//auto pRender2 = scn->addNode(std::make_shared<GLCommonPointVisualNode<DataType3f>>());


	mesh->varFileName()->setValue("../../data/standard/standard_sphere.obj");
	mesh->outTriangleSet()->connect(pickerNode->inTopology());
	/*mesh->outTriangleSet()->connect(surfacePickerNode->inInTopology());
	mesh->outTriangleSet()->connect(edgePickerNode->inInTopology());
	mesh->outTriangleSet()->connect(pointPickerNode->inInTopology());  */   

	/*edgePickerNode->varInterationRadius()->setValue(0.03f);
	pointPickerNode->varInterationRadius()->setValue(0.05f);*/
	pickerNode->varInterationRadius()->setValue(0.02f);
	/*surfacePickerNode->stateSelectedTopology()->connect(sRender1->inTriangleSet());
	sRender1->varColor()->setValue(Vec3f(0.2, 0.48, 0.75));
	surfacePickerNode->stateOtherTopology()->connect(sRender2->inTriangleSet());
	sRender2->varColor()->setValue(Vec3f(0.8, 0.52, 0.25));

	edgePickerNode->stateSelectedTopology()->connect(wRender1->inTriangleSet());
	wRender1->varColor()->setValue(Vec3f(0.8f, 0.0f, 0.0f));
	edgePickerNode->stateOtherTopology()->connect(wRender2->inTriangleSet());
	wRender2->varColor()->setValue(Vec3f(0.0f));

	pointPickerNode->stateSelectedTopology()->connect(pRender1->inPointSet());
	pRender1->varColor()->setValue(Vec3f(1.0f, 0, 0));
	pRender1->varPointSize()->setValue(0.015f);
	pointPickerNode->stateOtherTopology()->connect(pRender2->inPointSet());
	pRender2->varColor()->setValue(Vec3f(0, 0, 1.0f));
	pRender2->varPointSize()->setValue(0.01f);*/

	scn->setUpperBound({ 4, 4, 4 });

	QtApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}