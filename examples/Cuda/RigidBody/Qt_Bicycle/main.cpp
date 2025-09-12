#include <QtApp.h>

#include <SceneGraph.h>

#include <RigidBody/ConfigurableBody.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>
#include <Mapping/AnchorPointToPointSet.h>

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include <Module/GLPhotorealisticInstanceRender.h>

#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"

#include "GltfLoader.h"
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/MultibodySystem.h"
#include "RigidBody/Vehicle.h"
#include "RigidBody/Module/KeyDriver.h"


using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto bike = scn->addNode(std::make_shared<Bicycle<DataType3f>>());


	auto multisystem = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	auto driver = std::make_shared<KeyDriver<DataType3f>>();
	multisystem->stateTopology()->connect(driver->inTopology());
	multisystem->animationPipeline()->pushModule(driver);
	bike->outReset()->connect(driver->inReset());

	Key2HingeConfig keyConfig;
	//keyConfig.addMap(PKeyboardType::PKEY_W, 0, 1);
	//keyConfig.addMap(PKeyboardType::PKEY_S, 0, -1);

	keyConfig.addMap(PKeyboardType::PKEY_W, 1, 1);
	keyConfig.addMap(PKeyboardType::PKEY_S, 1, -1);

	keyConfig.addMap(PKeyboardType::PKEY_D, 2, 1);
	keyConfig.addMap(PKeyboardType::PKEY_A, 2, -1);
	driver->varHingeKeyConfig()->setValue(keyConfig);

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	bike->connect(multisystem->importVehicles());
	plane->stateTriangleSet()->connect(multisystem->inTriangleSet());
	plane->varLengthX()->setValue(120);
	plane->varLengthZ()->setValue(120);
	plane->varLocation()->setValue(Vec3f(0,-0.5,0));

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}
