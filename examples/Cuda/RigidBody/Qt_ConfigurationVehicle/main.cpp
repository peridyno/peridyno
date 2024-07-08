#include <QtApp.h>

#include <SceneGraph.h>

#include <RigidBody/Vechicle.h>

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

#include <PlaneModel.h>

#include "GltfLoader.h"
#include "ConvertToTextureMesh.h"
#include "CubeModel.h"
#include "CapsuleModel.h"
#include "SphereModel.h"
#include "GltfLoader.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	// 
	//auto convertBody = scn->addNode(std::make_shared<ConvertToTextureMesh<DataType3f>>());
	//cube->stateTriangleSet()->connect(convertBody->inTopology());
	// 
	//auto capsuleL = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	//auto convertL = scn->addNode(std::make_shared<ConvertToTextureMesh<DataType3f>>());
	//capsuleL->stateTriangleSet()->connect(convertL->inTopology());
	// 
	//auto capsuleR = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	//auto convertR = scn->addNode(std::make_shared<ConvertToTextureMesh<DataType3f>>());
	//capsuleR->stateTriangleSet()->connect(convertR->inTopology());

	auto configCar = scn->addNode(std::make_shared<ConfigurableVehicle<DataType3f>>());

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");
	gltf->setVisible(false);

	gltf->stateTextureMesh()->connect(configCar->inTextureMesh());


	VehicleBind configData;

	configData.vehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(std::string("LF"), 0), 0, Capsule));
	configData.vehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(std::string("LB"), 1), 1, Capsule));
	configData.vehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(std::string("RF"), 2), 2, Capsule));
	configData.vehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(std::string("RB"), 3), 3, Capsule));
	configData.vehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(std::string("BackWheel"), 4), 4, Box));
	configData.vehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(std::string("Body"), 5), 5, Box));

	configData.vehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(std::string("LF"), 0), Name_Shape(std::string("Body"), 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.vehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(std::string("LB"), 1), Name_Shape(std::string("Body"), 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.vehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(std::string("RF"), 2), Name_Shape(std::string("Body"), 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.vehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(std::string("RB"), 3), Name_Shape(std::string("Body"), 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));


	configCar->varVehicleConfiguration()->setValue(configData);

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


