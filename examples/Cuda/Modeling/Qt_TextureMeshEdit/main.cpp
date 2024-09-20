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
#include "TextureMeshMerge.h"
#include "ExtractShape.h"

/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */

int main()
{

	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(std::string(getAssetPath() + "Jeep/testEx.gltf"));

	auto extract = scn->addNode(std::make_shared<ExtractShape<DataType3f>>());

	auto merge = scn->addNode(std::make_shared<TextureMeshMerge<DataType3f>>());



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
	app.initialize(1280, 720);

	// setup envmap
	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
	}

	app.mainLoop();

	return 0;
}