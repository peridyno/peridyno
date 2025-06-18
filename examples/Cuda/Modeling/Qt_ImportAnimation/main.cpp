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
#include "ObjIO/ObjLoader.h"
#include "TextureMeshLoader.h"


/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */
using namespace dyno;
int main()
{

	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto gltfanimation = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltfanimation->varFileName()->setValue(getAssetPath() + "gltf/AnimationBone/Bone.gltf");
	gltfanimation->varImportAnimation()->setValue(true);
	gltfanimation->varUseInstanceTransform()->setValue(false);


	auto fbxAnimation = scn->addNode(std::make_shared<FBXLoader<DataType3f>>());
	fbxAnimation->varFileName()->setValue(getAssetPath() + "fbx/Dog.fbx");
	fbxAnimation->varUseInstanceTransform()->setValue(false);
	fbxAnimation->varImportAnimation()->setValue(true);
	fbxAnimation->varLocation()->setValue(Vec3f(-2, 0, 0));


	auto dancing = scn->addNode(std::make_shared<FBXLoader<DataType3f>>());
	dancing->varFileName()->setValue(getAssetPath() + "fbx/SwingDancing.fbx");
	dancing->varUseInstanceTransform()->setValue(false);
	dancing->varImportAnimation()->setValue(true);
	dancing->varLocation()->setValue(Vec3f(1, 0, 0));

	auto robot = scn->addNode(std::make_shared<FBXLoader<DataType3f>>());
	robot->varFileName()->setValue(getAssetPath() + "fbx/HumanoidRobot.fbx");
	robot->varUseInstanceTransform()->setValue(false);
	robot->varImportAnimation()->setValue(true);
	robot->varLocation()->setValue(Vec3f(2, 0, 0));

	auto bone = scn->addNode(std::make_shared<FBXLoader<DataType3f>>());
	bone->varFileName()->setValue(getAssetPath() + "fbx/Bone.fbx");
	bone->varUseInstanceTransform()->setValue(false);
	bone->varImportAnimation()->setValue(true);
	bone->varLocation()->setValue(Vec3f(-1, 0, 0));


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
		renderer->showGround = false;
	}

	app.mainLoop();

	return 0;
}