#include <QtApp.h>

//#include "TrianglePickerNode.h"
#include "initializeModeling.h"
#include "Commands/PointFromCurve.h"

#include "Commands/Sweep.h"
#include "BasicShapes/SphereModel.h"
#include "Commands/PolyExtrude.h"
#include "Commands/CopyToPoint.h"

#include "Group.h"

//ObjIO
#include "ObjIO/OBJexporter.h"
#include "ObjIO/ObjLoader.h"
#include "ObjIO/ObjPointLoader.h"


using namespace dyno;

int main()
{
	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	////Create Curve

	//auto sphere = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	//sphere->varScale()->setValue(Vec3f(0.2,0.2,0.2));
	//auto extrude = scn->addNode(std::make_shared<PolyExtrude<DataType3f>>());

	//auto picker = scn->addNode(std::make_shared<TrianglePickerNode<DataType3f>>());
	//picker->stateTriQuadIndex()->promoteOuput()->connect(extrude->inPrimId());

	//sphere->stateTriangleSet()->promoteOuput()->connect(extrude->inTriangleSet());

	//sphere->stateTriangleSet()->promoteOuput()->connect(picker->inTopology());
	
	////*************************************************************************************////

	auto obj = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());

	obj->varFileName()->setValue(getAssetPath() + "Building/YXH_Poly.obj");
	obj->varScale()->setValue(Vec3f(0.2,0.2,0.2));

	auto extrude = scn->addNode(std::make_shared<PolyExtrude<DataType3f>>());
	obj->outTriangleSet()->connect(extrude->inTriangleSet());
	extrude->varPrimitiveId()->setValue(" 0-109 292-413 430-836 1461-1486 1558-1647 1658-1709 1762-1842 1909-2132 2134-3016 3151-3154 3253-3326 3496 4816 4819 4828 5039 5956 7382-7383 7389-7392 7408 7413-7416 7722-7863 7871-7925 7935-8099 8102-8103 8140-8225 8245-8249 ");
	extrude->varDistance()->setValue(0.15);


	auto extrude2 = scn->addNode(std::make_shared<PolyExtrude<DataType3f>>());
	extrude->stateTriangleSet()->promoteOuput()->connect(extrude2->inTriangleSet());
	extrude2->varPrimitiveId()->setValue(" 837-1460 1487-1557 1648-1657 6422-6423 6438-6441 6483 6487-6488 6496 6519-6524 6595 6598-6606 6944 7654-7655 8126-8127");
	extrude2->varDistance()->setValue(0.3);

	auto extrude3 = scn->addNode(std::make_shared<PolyExtrude<DataType3f>>());
	extrude2->stateTriangleSet()->promoteOuput()->connect(extrude3->inTriangleSet());
	extrude3->varPrimitiveId()->setValue(" 110-290 ");
	extrude3->varDistance()->setValue(0.4);

	auto pt = scn->addNode(std::make_shared<ObjPoint<DataType3f>>());
	pt->varFileName()->setValue(getAssetPath()+"Building/Tree_Scatter.obj");

	auto tree = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	tree->varFileName()->setValue(getAssetPath() + "Building/Tree_Poly.obj");

	auto copy = scn->addNode(std::make_shared<CopyToPoint<DataType3f>>());
	tree->outTriangleSet()->connect(copy->inTriangleSetIn());
	pt->outPointSet()->promoteOuput()->connect(copy->inTargetPointSet());


	////*************************************************************************************////


	auto group = scn->addNode(std::make_shared<Group<DataType3f>>());
	group->varPrimitiveId()->setValue(" 1 2-8 19-25");
	group->varEdgeId()->setValue(" 3-8 12 16 25-27");
	group->varPointId()->setValue(" 10 15-20 30 35 38-40");


	Modeling::initStaticPlugin();


	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}