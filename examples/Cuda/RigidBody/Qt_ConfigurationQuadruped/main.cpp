#include <QtApp.h>

#include <SceneGraph.h>
#include <HeightField/GranularMedia.h>
#include <BasicShapes/PlaneModel.h>
#include <BasicShapes/PlaneModel.h>

#include <RigidBody/ConfigurableBody.h>
#include <RigidBody/Module/CarDriver.h>

#include "BasicShapes/PlaneModel.h"
#include "FBXLoader/FBXLoader.h"
#include "RigidBody/Module/AnimationDriver.h"
#include "RigidBody/MultibodySystem.h"
#include <HeightField/SurfaceParticleTracking.h>
#include <HeightField/RigidSandCoupling.h>

using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto fbx = scn->addNode(std::make_shared<FBXLoader<DataType3f>>());
	fbx->varImportAnimation()->setValue(true);
	fbx->varFileName()->setValue(getAssetPath() + "fbx/Dog.fbx");
	fbx->reset();
	fbx->setVisible(false);

	auto robot = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());
	fbx->stateTextureMesh()->connect(robot->inTextureMesh());
	robot->varLocation()->setValue(Vec3f(0,0.3,0));

	std::string body = std::string("Model::Robot_GLTF:Body");
	std::string lf_up = std::string("Model::Robot_GLTF:LF_Up");
	std::string lf_down = std::string("Model::Robot_GLTF:LF_Down");
	std::string lb_up = std::string("Model::Robot_GLTF:LB_Up");
	std::string lb_down = std::string("Model::Robot_GLTF:LB_Down");
	std::string rf_up = std::string("Model::Robot_GLTF:RF_Up");
	std::string rf_down = std::string("Model::Robot_GLTF:RF_Down");
	std::string rb_up = std::string("Model::Robot_GLTF:RB_Up");
	std::string rb_down = std::string("Model::Robot_GLTF:RB_Down");

	MultiBodyTuple multiBodyConfig;
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(body, 0, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(body), RigidShapeType::SHAPE_BOX, 5100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(lf_up, 1, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(lf_up), RigidShapeType::SHAPE_BOX, 5100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(lf_down, 2, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(lf_down), RigidShapeType::SHAPE_BOX, 5100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(lb_up, 3, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(lb_up), RigidShapeType::SHAPE_BOX, 5100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(lb_down, 4, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(lb_down), RigidShapeType::SHAPE_BOX, 5100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(rf_up, 5, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(rf_up), RigidShapeType::SHAPE_BOX, 5100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(rf_down, 6, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(rf_down), RigidShapeType::SHAPE_BOX, 5100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(rb_up, 7, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(rb_up), RigidShapeType::SHAPE_BOX, 5100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(rb_down, 8, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(rb_down), RigidShapeType::SHAPE_BOX, 5100));

	for (auto it = multiBodyConfig.varRigidBodyConfigs()->begin(); it != multiBodyConfig.varRigidBodyConfigs()->end(); it++)
	{
		auto rigid = multiBodyConfig.varRigidBodyConfigs()->getElement(it);
		auto shape = rigid.varShapeConfigs()->getElement(rigid.varShapeConfigs()->begin());
		shape.varRadius()->setValue(0.2);
	}

	Vec3f offset = Vec3f(0, 0.17, 0);
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(lf_up, 1, body, 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), offset, true, 0));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(lf_down, 2, lf_up, 1, JointType::JOINT_Hinge, Vec3f(1, 0, 0), offset, true, 0));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(lb_up, 3, body, 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), offset, true, 0));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(lb_down, 4, lb_up, 3, JointType::JOINT_Hinge, Vec3f(1, 0, 0), offset, true, 0));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(rf_up, 5, body, 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), offset, true, 0));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(rf_down, 6, rf_up, 5, JointType::JOINT_Hinge, Vec3f(1, 0, 0), offset, true, 0));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(rb_up, 7, body, 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), offset, true, 0));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(rb_down, 8, rb_up, 7, JointType::JOINT_Hinge, Vec3f(1, 0, 0), offset, true, 0));

	robot->varConfiguration()->setValue(multiBodyConfig);

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	multibody->varFrictionCoefficient()->setValue(200);
	multibody->varGravityValue()->setValue(9.8);

	robot->connect(multibody->importVehicles());

	auto animDriver = std::make_shared<AnimationDriver<DataType3f>>();
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::LFU_2"), 0, 2, 0.5));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::LFD_3"), 1, 2, 0.5));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::LBU_5"), 2, 2, 0.5));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::LBD_6"), 3, 2, 0.5));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::RFU_8"), 4, 2, 0.5));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::RFD_9"), 5, 2, 0.5));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::RBU_11"), 6, 2, 0.5));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::RBD_12"), 7, 2, 0.5));

	animDriver->varSpeed()->setValue(8);
	fbx->stateJointAnimationInfo()->connect(animDriver->inJointAnimationInfo());

	multibody->animationPipeline()->pushModule(animDriver);
	multibody->stateTimeStep()->connect(animDriver->inDeltaTime());
	multibody->stateTopology()->connect(animDriver->inTopology());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(20));
	plane->varSegmentX()->setValue(5);
	plane->varSegmentZ()->setValue(5);
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());

	//float spacing = 0.1f;
	//uint res = 256;
	//auto sand = scn->addNode(std::make_shared<GranularMedia<DataType3f>>());
	//sand->varOrigin()->setValue(-0.5f * Vec3f(res * spacing, 0.0f, res * spacing));
	//sand->varSpacing()->setValue(spacing);
	//sand->varWidth()->setValue(res);
	//sand->varHeight()->setValue(res);
	//sand->varDepth()->setValue(0.15);
	//sand->varDepthOfDiluteLayer()->setValue(0.1);


	//auto coupling = scn->addNode(std::make_shared<RigidSandCoupling<DataType3f>>());
	//multibody->connect(coupling->importRigidBodySystem());
	//sand->connect(coupling->importGranularMedia());

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


