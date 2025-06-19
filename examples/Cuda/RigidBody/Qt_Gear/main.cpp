#include <UbiApp.h>

#include <SceneGraph.h>

#include <RigidBody/Gear.h>
#include <RigidBody/MultibodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include <BasicShapes/PlaneModel.h>

#include <Collision/NeighborElementQuery.h>

#include "RigidBody/Vehicle.h"

using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> createSceneGraph()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));


	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(50);
	plane->varLengthZ()->setValue(50);
	plane->varSegmentX()->setValue(10);
	plane->varSegmentZ()->setValue(10);

	auto convoy = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());

	for (int i = 0; i < 5; i++)
	{
		auto gear = scn->addNode(std::make_shared<Bug<DataType3f>>());
		gear->setfile(std::string("ma/bigring"));
		gear->setposition(Vec3f(0, 10 + 10.0 * i, 0 + 0.4 * i));
		gear->setv(Vec3f(10 * i, 0, 0));
		gear->connect(convoy->importVehicles());
	}

	for (int j = 0; j < 5; j++)
	{
		auto gear = scn->addNode(std::make_shared<Bug<DataType3f>>());
		gear->setfile(std::string("ma/bigbug"));
		gear->setposition(Vec3f(0 +j * 20.0, 3, 0));
		gear->connect(convoy->importVehicles());
	}

	plane->stateTriangleSet()->connect(convoy->inTriangleSet());

	return scn;
}

int main()
{
	/*UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createSceneGraph());

	app.initialize(1280, 768);
	app.renderWindow()->getCamera()->setUnitScale(3.0f);
	app.mainLoop();

	return 0;*/
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createSceneGraph());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}