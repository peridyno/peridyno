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

#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"


#include "GltfLoader.h"
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/Module/QuadrupedDriver.h"
#include "RigidBody/Module/CarDriver.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto configCar = scn->addNode(std::make_shared<ConfigurableVehicle<DataType3f>>());

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "gltf/Quadruped/Quadruped.gltf");
	gltf->setVisible(false);

	gltf->stateTextureMesh()->connect(configCar->inTextureMesh());


	VehicleBind configData;

	Vec3f angle = Vec3f(0,0,90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("Body", 0), 0, Box, Transform3f(),0.1));//
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("LF_Up", 1), 1, Box, Transform3f(),100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("LF_Down", 2), 2, Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("LB_Up", 3), 3, Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("LB_Down", 4), 4, Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("RF_Up", 5),5, Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("RF_Down", 6), 6, Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("RB_Up", 7), 7, Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("RB_Down", 8), 8, Box, Transform3f(), 100));

	for (size_t i = 0; i < configData.mVehicleRigidBodyInfo.size(); i++)
	{
		configData.mVehicleRigidBodyInfo[i].radius = 0.2;
	}


	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LF_Up", 1), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0,0.2,0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LF_Down", 2), Name_Shape("LF_Up", 1), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LB_Up", 3), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LB_Down", 4), Name_Shape("LB_Up", 3), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RF_Up", 5), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RF_Down", 6), Name_Shape("RF_Up", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RB_Up", 7), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RB_Down", 8), Name_Shape("RB_Up", 7), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LF_Up", 1), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LF_Down", 2), Name_Shape("LF_Up", 1), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LB_Up", 3), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LB_Down", 4), Name_Shape("LB_Up", 3), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RF_Up", 5), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RF_Down", 6), Name_Shape("RF_Up", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RB_Up", 7), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RB_Down", 8), Name_Shape("RB_Up", 7), Hinge, Vec3f(1, 0, 0), Vec3f(0, 0.2, 0), true, 0));

	configCar->varVehicleConfiguration()->setValue(configData); 
	auto carDriver = configCar->animationPipeline()->findFirstModule<CarDriver<DataType3f>>();
	configCar->animationPipeline()->popModule(carDriver);

	
	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(50);
	plane->varLengthZ()->setValue(50);

	plane->stateTriangleSet()->connect(configCar->inTriangleSet());

	auto quadDrive = std::make_shared<QuadrupedDriver<DataType3f>>();
	configCar->stateTimeStep()->connect(quadDrive->inDeltaTime());
	configCar->stateTopology()->connect(quadDrive->inTopology());
	configCar->animationPipeline()->pushModule(quadDrive);
	//configCar->varGravityValue()->setValue(0);

	std::vector<Transform3f> vehicleTransforms;

	vehicleTransforms.push_back(Transform3f(Vec3f(0,1,0),Quat1f(0,Vec3f(0,0.1,0)).toMatrix3x3()));


	configCar->varVehiclesTransform()->setValue(vehicleTransforms);

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


