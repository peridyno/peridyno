#include <QtApp.h>

#include "RigidBody/initializeRigidBody.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "Peridynamics/initializePeridynamics.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "Volume/initializeVolume.h"
#include "Multiphysics/initializeMultiphysics.h"
#include "initializeModeling.h"
#include "initializeIO.h"
#include "GltfLoader.h"
#include <GLRenderEngine.h>
#include "JointDeform.h"
#include "FBXLoader/FBXLoader.h"
#include "RigidBody/Module/AnimationDriver.h"
#include "RigidBody/Module/CarDriver.h"
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/MultibodySystem.h"

#include "RigidBody/ConfigurableBody.h"


/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */
using namespace dyno;
int main()
{

	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto fbx = scn->addNode(std::make_shared<FBXLoader<DataType3f>>());
	fbx->varFileName()->setValue(getAssetPath() + "fbx/HumanoidRobot.fbx");
	fbx->varImportAnimation()->setValue(true);
	fbx->reset();

	auto robot = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());
	fbx->stateTextureMesh()->connect(robot->inTextureMesh());
	fbx->setVisible(false);


	MultiBodyTuple configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);

	std::string Hip = std::string("Model::Hip");

	std::string Trochanter_R = std::string("Model::Trochanter_R");
	std::string Thigh_R = std::string("Model::Thigh_R");
	std::string Shank_R = std::string("Model::Shank_R");
	std::string Foot_R = std::string("Model::Foot_R");

	std::string Trochanter_L = std::string("Model::Trochanter_L");
	std::string Thigh_L = std::string("Model::Thigh_L");
	std::string Shank_L = std::string("Model::Shank_L");
	std::string Foot_L = std::string("Model::Foot_L");

	std::string Spine = std::string("Model::Spine");
	std::string Body = std::string("Model::Body");
	std::string Neck = std::string("Model::Neck1");
	std::string Head = std::string("Model::Head");

	std::string Shoulder_R = std::string("Model::Shoulder_R");
	std::string UpperArm_R = std::string("Model::UpperArm_R");
	std::string LowerArm_R = std::string("Model::LowerArm_R");
	std::string Wrist_R = std::string("Model::Wrist_R");
	std::string Hand_R = std::string("Model::Hand_R");

	std::string Shoulder_L = std::string("Model::Shoulder_L");
	std::string UpperArm_L = std::string("Model::UpperArm_L");
	std::string LowerArm_L = std::string("Model::LowerArm_L");
	std::string Wrist_L = std::string("Model::Wrist_L");
	std::string Hand_L = std::string("Model::Hand_L");

	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Hip, 0, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hip), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Trochanter_R, 1, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Trochanter_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Thigh_R, 2, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Thigh_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Shank_R, 3, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shank_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Foot_R, 4, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Foot_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Trochanter_L, 5, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Trochanter_L), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Thigh_L, 6, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Thigh_L), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Shank_L, 7, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shank_L), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Foot_L, 8, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Foot_L), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Spine, 9, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Spine), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Body, 10, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Body), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Neck, 11, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Neck), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Head, 12, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Head), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Shoulder_R, 13, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shoulder_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(UpperArm_R, 14, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(UpperArm_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(LowerArm_R, 15, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(LowerArm_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Wrist_R, 16, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Wrist_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Hand_R, 17, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hand_R), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Shoulder_L, 18, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shoulder_L), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(UpperArm_L, 19, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(UpperArm_L), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(LowerArm_L, 20, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(LowerArm_L), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Wrist_L, 21, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Wrist_L), RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple(Hand_L, 22, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hand_L), RigidShapeType::SHAPE_BOX, 100));


	Vec3f offset = Vec3f(0, 0, 0);
	Vec3f shankOffset = Vec3f(0, 0.25, 0);
	Vec3f thighOffset = Vec3f(0, 0.28, 0);
	Vec3f footOffset = Vec3f(0, 0, 0.02);
	Vec3f bodyOffset = Vec3f(0, -0.25, 0);
	Vec3f handOffset = Vec3f(0.02, 0, 0);

	Vec3f axis = Vec3f(1,0,0);

	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Trochanter_R, 1, Hip, 0, JointType::JOINT_Hinge, axis, offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Trochanter_L, 5, Hip, 0, JointType::JOINT_Hinge, axis, offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Shank_R, 3, Thigh_R, 2, JointType::JOINT_Hinge, axis, shankOffset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Shank_L, 7, Thigh_L, 6, JointType::JOINT_Hinge, axis, shankOffset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Foot_R, 4, Shank_R, 3, JointType::JOINT_Hinge, axis, footOffset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Foot_L, 8, Shank_L, 7, JointType::JOINT_Hinge, axis, footOffset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Thigh_R, 2, Trochanter_R, 1, JointType::JOINT_Hinge, axis, thighOffset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Thigh_L, 6, Trochanter_L, 5, JointType::JOINT_Hinge, axis, thighOffset, true, 0, true, -90, 90));
	
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Spine, 9, Hip, 0, JointType::JOINT_Fixed, Vec3f(0, 1, 0), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Body, 10, Spine, 9, JointType::JOINT_Fixed, Vec3f(1, 0, 0), bodyOffset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Neck, 11, Body, 10, JointType::JOINT_Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Head, 12, Neck, 11, JointType::JOINT_Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Shoulder_R, 13, Body, 10, JointType::JOINT_Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Shoulder_L, 18, Body, 10, JointType::JOINT_Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(UpperArm_R, 14, Shoulder_R, 13, JointType::JOINT_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(UpperArm_L, 19, Shoulder_L, 18, JointType::JOINT_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(LowerArm_R, 15, UpperArm_R, 14, JointType::JOINT_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(LowerArm_L, 20, UpperArm_L, 19, JointType::JOINT_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Wrist_R, 16, LowerArm_R, 15, JointType::JOINT_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Wrist_L, 21, LowerArm_L, 20, JointType::JOINT_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Hand_R, 17, Wrist_R, 16, JointType::JOINT_Fixed, Vec3f(1, 0, 0), handOffset, true, 0, true, -90, 90));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple(Hand_L, 22, Wrist_L, 21, JointType::JOINT_Fixed, Vec3f(1, 0, 0), -handOffset, true, 0, true, -90, 90));

	
	robot->varConfiguration()->setValue(configData);

	robot->varGravityValue()->setValue(0);
	
	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	robot->connect(multibody->importVehicles());

	auto animDriver = std::make_shared<AnimationDriver<DataType3f>>();
	multibody->animationPipeline()->pushModule(animDriver);

	float weight = 1;
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::R_Hip01"), 0, 1, weight));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::L_Hip01"), 1, 1, weight));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::R_Knee"), 2, 1, weight));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::L_Knee"), 3, 1, weight));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::R_Foot"), 4, 1, weight));
	animDriver->varBindingConfiguration()->pushBack(Animation2JointConfigTuple(std::string("Model::L_Foot"), 5, 1, weight));

	animDriver->varSpeed()->setValue(1);
	multibody->stateTimeStep()->connect(animDriver->inDeltaTime());
	fbx->stateJointAnimationInfo()->connect(animDriver->inJointAnimationInfo());
	multibody->stateTopology()->connect(animDriver->inTopology());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(20));
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());


	Modeling::initStaticPlugin();
	RigidBody::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	Peridynamics::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();
	Volume::initStaticPlugin();
	Multiphysics::initStaticPlugin();
	dynoIO::initStaticPlugin();


	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1280	, 720);

	// setup envmap
	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
	}

	app.mainLoop();

	return 0;
}