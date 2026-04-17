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


	MultiBodyBind configData;

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


	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Hip, 0), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hip), CONFIG_BOX,100));//
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Trochanter_R, 1), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Trochanter_R), CONFIG_BOX, 100));//
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Thigh_R, 2), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Thigh_R), CONFIG_BOX, 100));//
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Shank_R, 3), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shank_R), CONFIG_BOX, 100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Foot_R, 4), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Foot_R), CONFIG_BOX, 100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Trochanter_L, 5), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Trochanter_L), CONFIG_BOX, 100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Thigh_L, 6), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Thigh_L), CONFIG_BOX, 100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Shank_L, 7), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shank_L), CONFIG_BOX, 100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Foot_L, 8), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Foot_L), CONFIG_BOX, 100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Spine, 9), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Spine), CONFIG_BOX,  100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Body, 10), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Body), CONFIG_BOX,  100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Neck, 11), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Neck), CONFIG_BOX,  100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Head, 12), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Head), CONFIG_BOX,  100));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Shoulder_R, 13), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shoulder_R), CONFIG_BOX,  20));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(UpperArm_R, 14), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(UpperArm_R), CONFIG_BOX,  40));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(LowerArm_R, 15), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(LowerArm_R), CONFIG_BOX,  40));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Wrist_R, 16), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Wrist_R), CONFIG_BOX,  10));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Hand_R, 17), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hand_R), CONFIG_BOX,  10));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Shoulder_L, 18), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shoulder_L), CONFIG_BOX,  20));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(UpperArm_L, 19), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(UpperArm_L), CONFIG_BOX,  40));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(LowerArm_L, 20), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(LowerArm_L), CONFIG_BOX,  40));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Wrist_L, 21), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Wrist_L), CONFIG_BOX,  10));
	configData. rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID(Hand_L, 22), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hand_L), CONFIG_BOX,  10));


	Vec3f offset = Vec3f(0, 0, 0);
	Vec3f shankOffset = Vec3f(0, 0.25, 0);
	Vec3f thighOffset = Vec3f(0, 0.28, 0);
	Vec3f footOffset = Vec3f(0, 0, 0.02);
	Vec3f bodyOffset = Vec3f(0, -0.25, 0);
	Vec3f handOffset = Vec3f(0.02, 0, 0);

	Vec3f axis = Vec3f(1,0,0);

	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Trochanter_R, 1), NameRigidID(Hip, 0), CONFIG_Hinge, axis, offset, true, 0,true,-90,90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Trochanter_L, 5), NameRigidID(Hip, 0), CONFIG_Hinge, axis, offset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Shank_R, 3), NameRigidID(Thigh_R, 2), CONFIG_Hinge, axis, shankOffset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Shank_L, 7), NameRigidID(Thigh_L, 6), CONFIG_Hinge, axis, shankOffset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Foot_R, 4), NameRigidID(Shank_R, 3), CONFIG_Hinge, axis, footOffset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Foot_L, 8), NameRigidID(Shank_L, 7), CONFIG_Hinge, axis, footOffset, true, 0, true, -90, 90));


	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Thigh_R, 2), NameRigidID(Trochanter_R, 1), CONFIG_Fixed, axis, thighOffset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Thigh_L, 6), NameRigidID(Trochanter_L, 5), CONFIG_Fixed, axis, thighOffset, true, 0, true, -90, 90));

	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Spine, 9), NameRigidID(Hip, 0), CONFIG_Fixed, Vec3f(0, 1, 0), offset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Body, 10), NameRigidID(Spine, 9), CONFIG_Fixed, Vec3f(1, 0, 0), bodyOffset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Neck, 11), NameRigidID(Body, 10), CONFIG_Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Head, 12), NameRigidID(Neck, 11), CONFIG_Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Shoulder_R, 13), NameRigidID(Body, 10), CONFIG_Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Shoulder_L, 18), NameRigidID(Body, 10), CONFIG_Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(UpperArm_R, 14), NameRigidID(Shoulder_R, 13), CONFIG_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(UpperArm_L, 19), NameRigidID(Shoulder_L, 18), CONFIG_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));

	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(LowerArm_R, 15), NameRigidID(UpperArm_R, 14), CONFIG_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(LowerArm_L, 20), NameRigidID(UpperArm_L, 19), CONFIG_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));

	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Wrist_R, 16), NameRigidID(LowerArm_R, 15), CONFIG_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Wrist_L, 21), NameRigidID(LowerArm_L, 20), CONFIG_Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Hand_R, 17), NameRigidID(Wrist_R, 16), CONFIG_Fixed, Vec3f(1, 0, 0), handOffset, true, 0, true, -90, 90));
	configData. jointConfigs.push_back(MultiBodyJointConfig(NameRigidID(Hand_L, 22), NameRigidID(Wrist_L, 21), CONFIG_Fixed, Vec3f(1, 0, 0), -handOffset, true, 0, true, -90, 90));

	robot->varConfiguration()->setValue(configData);

	robot->varGravityValue()->setValue(0);
	
	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	robot->connect(multibody->importVehicles());

	auto animDriver = std::make_shared<AnimationDriver<DataType3f>>();
	multibody->animationPipeline()->pushModule(animDriver);


	std::vector<Animation2JointConfig> config(6);

	float weight = 1;

	config[0] = Animation2JointConfig(std::string("Model::R_Hip01"), 0, 1, weight);
	config[1] = Animation2JointConfig(std::string("Model::L_Hip01"), 1, 1, weight);

	config[2] = Animation2JointConfig(std::string("Model::R_Knee"), 2, 1, weight);
	config[3] = Animation2JointConfig(std::string("Model::L_Knee"), 3, 1, weight);

	config[4] = Animation2JointConfig(std::string("Model::R_Foot"), 4, 1, weight);
	config[5] = Animation2JointConfig(std::string("Model::L_Foot"), 5, 1, weight);


	animDriver->varBindingConfiguration()->setValue(config);
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