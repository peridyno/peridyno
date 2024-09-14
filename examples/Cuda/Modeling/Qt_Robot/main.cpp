#include <QtApp.h>
using namespace dyno;

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
#include "AnimationMixer.h"

#include "RigidBody/SkeletonRigidBody.h"

#include "Ik.h"


/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */

int main()
{

	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto idle = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	idle->varFileName()->setValue(getAssetPath()+std::string("gltf/Character/Idle.gltf"));//std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf")
	idle->varUseInstanceTransform()->setValue(false);
	idle->varImportAnimation()->setValue(true);
	idle->setVisible(false);

	auto walk = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	walk->varFileName()->setValue(getAssetPath() + std::string("gltf/Character/Walk.gltf"));//std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf")
	walk->varUseInstanceTransform()->setValue(false);
	walk->varImportAnimation()->setValue(true);
	walk->setVisible(false);

	auto run = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	run->varFileName()->setValue(getAssetPath() + std::string("gltf/Character/Run.gltf"));//std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf")
	run->varUseInstanceTransform()->setValue(false);
	run->varImportAnimation()->setValue(true);
	run->setVisible(false);

	auto mixer = scn->addNode(std::make_shared<AnimationMixer<DataType3f>>());
	idle->stateAnimation()->connect(mixer->inIdle());
	idle->stateTextureMesh()->connect(mixer->inTextureMesh());
	walk->stateAnimation()->connect(mixer->inWalk());

	auto skeletonBody = scn->addNode(std::make_shared<SkeletonRigidBody<DataType3f>>());
	mixer->stateElementsCenter()->promoteOuput()->connect(skeletonBody->inElementsCenter());
	mixer->stateElementsQuaternion()->promoteOuput()->connect(skeletonBody->inElementsQuaternion());
	mixer->stateElementsRotationMatrix()->promoteOuput()->connect(skeletonBody->inElementsRotation());
	mixer->stateElementsLength()->promoteOuput()->connect(skeletonBody->inElementsLength());
	mixer->stateElementsRadius()->promoteOuput()->connect(skeletonBody->inElementsRadius());


	idle->stateJointsData()->promoteOuput()->connect(mixer->inSkeleton());

	auto deformer = scn->addNode(std::make_shared<JointDeform<DataType3f>>());
	idle->stateSkin()->connect(deformer->inSkin());
	mixer->stateJoint()->connect(deformer->inJoint());
	mixer->stateInstanceTransform()->connect(deformer->inInstanceTransform());


	// hard code
	idle->stateTextureMesh()->getData().materials()[0]->metallic = 1.f;
	idle->stateTextureMesh()->getData().materials()[0]->roughness = 0.15f;


	////Jump
	// 
	//auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	//gltf->varFileName()->setValue(getAssetPath() + std::string("gltf/Character/Character_Jump_26F.gltf"));//std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf")
	//gltf->varUseInstanceTransform()->setValue(false);
	//gltf->varImportAnimation()->setValue(true);
	//gltf->setVisible(false);

	//auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	//gltf->varFileName()->setValue(getAssetPath() + std::string("gltf/Character/Character_Land_26F.gltf"));//std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf")
	//gltf->varUseInstanceTransform()->setValue(false);
	//gltf->varImportAnimation()->setValue(true);
	//gltf->setVisible(false);

	//auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	//gltf->varFileName()->setValue(getAssetPath() + std::string("gltf/Character/Character_Loop_90F.gltf"));//std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf")
	//gltf->varUseInstanceTransform()->setValue(false);
	//gltf->varImportAnimation()->setValue(true);
	//gltf->setVisible(false);




	//auto gltf2 = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	//gltf2->varFileName()->setValue(std::string("C:/Users/dell/Desktop/RobotRun_V2.gltf"));//std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf")
	//gltf2->varUseInstanceTransform()->setValue(false);
	//gltf2->varImportAnimation()->setValue(true);
	//gltf2->setVisible(false);

	//auto jointDeform = scn->addNode(std::make_shared<JointDeform<DataType3f>>());
	//gltf->stateJointsData()->connect(jointDeform->inJoint());
	//gltf->stateSkin()->connect(jointDeform->inSkin());
	//


	//gltf->stateAnimation()->connect(Mixer->inAnimation01());
	//gltf->stateAnimation()->connect(Mixer->inAnimation02());
	//gltf->stateJointsData()->connect(Mixer->inSkeleton());

	//Mixer->stateJoint()->connect(jointDeform->inJoint());



	//auto ik = scn->addNode(std::make_shared<IK<DataType3f>>());


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