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

/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */

int main()
{

	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath()+std::string("gltf/Robot/RobotTest.gltf"));//std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf")
	gltf->varUseInstanceTransform()->setValue(false);
	gltf->varImportAnimation()->setValue(true); 
	gltf->setVisible(false);

	auto jointDeform = scn->addNode(std::make_shared<JointDeform<DataType3f>>());
	gltf->stateJointsData()->connect(jointDeform->inJoint());
	gltf->stateSkin()->connect(jointDeform->inSkin());



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