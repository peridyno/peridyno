#include <QtApp.h>

#include <SceneGraph.h>

#include <BasicShapes/PlaneModel.h>
#include <BasicShapes/PlaneModel.h>

#include <RigidBody/ConfigurableBody.h>
#include <RigidBody/Module/CarDriver.h>

#include "BasicShapes/PlaneModel.h"
#include "SkeletonLoader/SkeletonLoader.h"
#include "RigidBody/Module/AnimationDriver.h"
#include "RigidBody/MultibodySystem.h"


using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto fbx = scn->addNode(std::make_shared<SkeletonLoader<DataType3f>>());
	fbx->varFileName()->setValue(getAssetPath() + "fbx/Dog.fbx");
	fbx->reset();
	fbx->setVisible(false);

	auto robot = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());
	fbx->stateTextureMesh()->connect(robot->inTextureMesh());

	VehicleBind configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);

	std::string body = std::string("Model::Robot_GLTF:Body");
	std::string lf_up = std::string("Model::Robot_GLTF:LF_Up");
	std::string lf_down = std::string("Model::Robot_GLTF:LF_Down");
	std::string lb_up = std::string("Model::Robot_GLTF:LB_Up");
	std::string lb_down = std::string("Model::Robot_GLTF:LB_Down");
	std::string rf_up = std::string("Model::Robot_GLTF:RF_Up");
	std::string rf_down = std::string("Model::Robot_GLTF:RF_Down");
	std::string rb_up = std::string("Model::Robot_GLTF:RB_Up");
	std::string rb_down = std::string("Model::Robot_GLTF:RB_Down");

	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(body, 0), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(body), Box, Transform3f(), 100));//
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(lf_up, 1), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(lf_up), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(lf_down, 2), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(lf_down), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(lb_up, 3), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(lb_up), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(lb_down, 4), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(lb_down), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(rf_up, 5), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(rf_up), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(rf_down, 6), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(rf_down), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(rb_up, 7), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(rb_up), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(rb_down, 8), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(rb_down), Box, Transform3f(), 100));

	for (size_t i = 0; i < configData.mVehicleRigidBodyInfo.size(); i++)
	{
		configData.mVehicleRigidBodyInfo[i].radius = 0.2;
	}

	Vec3f offset = Vec3f(0, 0.17, 0);

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(lf_up, 1), Name_Shape(body, 0), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(lf_down, 2), Name_Shape(lf_up, 1), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(lb_up, 3), Name_Shape(body, 0), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(lb_down, 4), Name_Shape(lb_up, 3), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(rf_up, 5), Name_Shape(body, 0), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(rf_down, 6), Name_Shape(rf_up, 5), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(rb_up, 7), Name_Shape(body, 0), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(rb_down, 8), Name_Shape(rb_up, 7), Hinge, Vec3f(1, 0, 0), offset, true, 0));

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(lf_up, 1), Name_Shape(body, 0), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(lf_down, 2), Name_Shape(lf_up, 1), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(lb_up, 3), Name_Shape(body, 0), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(lb_down, 4), Name_Shape(lb_up, 3), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(rf_up, 5), Name_Shape(body, 0), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(rf_down, 6), Name_Shape(rf_up, 5), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(rb_up, 7), Name_Shape(body, 0), Hinge, Vec3f(1, 0, 0), offset, true, 0));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(rb_down, 8), Name_Shape(rb_up, 7), Hinge, Vec3f(1, 0, 0), offset, true, 0));


	robot->varVehicleConfiguration()->setValue(configData);

	std::vector<std::string> hinge_DriveObjName(configData.mVehicleJointInfo.size(), "NULL");

	hinge_DriveObjName[0] = std::string("Model::LFU_2");
	hinge_DriveObjName[1] = std::string("Model::LFD_3");
	hinge_DriveObjName[2] = std::string("Model::LBU_5");
	hinge_DriveObjName[3] = std::string("Model::LBD_6");
	hinge_DriveObjName[4] = std::string("Model::RFU_8");
	hinge_DriveObjName[5] = std::string("Model::RFD_9");
	hinge_DriveObjName[6] = std::string("Model::RBU_11");
	hinge_DriveObjName[7] = std::string("Model::RBD_12");

	hinge_DriveObjName[8] = std::string("Model::LFU_2");
	hinge_DriveObjName[9] = std::string("Model::LFD_3");
	hinge_DriveObjName[10] = std::string("Model::LBU_5");
	hinge_DriveObjName[11] = std::string("Model::LBD_6");
	hinge_DriveObjName[12] = std::string("Model::RFU_8");
	hinge_DriveObjName[13] = std::string("Model::RFD_9");
	hinge_DriveObjName[14] = std::string("Model::RBU_11");
	hinge_DriveObjName[15] = std::string("Model::RBD_12");


	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	robot->connect(multibody->importVehicles());

	auto animDriver = std::make_shared<AnimationDriver<DataType3f>>();
	animDriver->varDriverName()->setValue(hinge_DriveObjName);

	multibody->animationPipeline()->pushModule(animDriver);
	multibody->stateTimeStep()->connect(animDriver->inDeltaTime());
	fbx->stateHierarchicalScene()->connect(animDriver->inHierarchicalScene());
	multibody->stateTopology()->connect(animDriver->inTopology());
	animDriver->varSpeed()->setValue(1);

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(20));
	plane->varSegmentX()->setValue(5);
	plane->varSegmentZ()->setValue(5);
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());


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


