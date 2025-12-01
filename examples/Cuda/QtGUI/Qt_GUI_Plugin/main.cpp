#include <QtApp.h>
using namespace dyno;

#include "RigidBody/initializeRigidBody.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "DualParticleSystem/initializeDualParticleSystem.h"
#include "Peridynamics/initializePeridynamics.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "Volume/initializeVolume.h"
#include "Multiphysics/initializeMultiphysics.h"
#include "HeightField/initializeHeightField.h"
#include "initializeModeling.h"
#include "initializeIO.h"
#include "Topology/MaterialManager.h"

#include "Break_MakeTextureMesh.h"
#include "GLPhotorealisticRender.h"
#include "ImageLoader.h"
#include "GltfLoader.h"
#include "GLPhotorealisticRender.h"

/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */

int main()
{
	MaterialManager::NewMaterial();
	MaterialManager::createCustomMaterial();

	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(std::string(getAssetPath() + "gltf/test/sphere.gltf"));

	//auto srcMaterial = MaterialManager::getMaterial("BodyMaterial");
	//auto customMaterial = MaterialManager::createCustomMaterial(srcMaterial);
	//gltf->graphicsPipeline()->pushModule(srcMaterial);

	//auto matPipeline = customMaterial->materialPipeline();
	//gltf->graphicsPipeline()->pushModule(customMaterial);

	//auto texCorrect = std::make_shared<ColorCorrect>();
	//srcMaterial->outTexColor()->connect(texCorrect->inTexture());
	//texCorrect->varSaturation()->setValue(0);
	//texCorrect->outTexture()->connect(customMaterial->inTexColor());
	//matPipeline->pushModule(texCorrect);
	//gltf->graphicsPipeline()->pushModule(texCorrect);

	//auto image = std::make_shared<ImageLoaderModule>();
	//image->varImagePath()->setValue(std::string(getAssetPath() + "Jeep/JeepGltf/jeep_body_camouflage.png"));
	//matPipeline->pushModule(image);

	//matPipeline->updateMaterialPipline();
	//gltf->graphicsPipeline()->pushModule(image);

	//auto assignMaterial = std::make_shared<AssignTextureMeshMaterial<DataType3f>>();
	//assignMaterial->varShapeIndex()->setValue(5);
	//assignMaterial->varMaterialName()->setValue(customMaterial->getName());
	//auto textureRender = gltf->graphicsPipeline()->findFirstModule<GLPhotorealisticRender>();
	//if (textureRender)
	//	assignMaterial->outTextureMesh()->connect(textureRender->inTextureMesh());
	//gltf->stateTextureMesh()->connect(assignMaterial->inTextureMesh());
	//gltf->graphicsPipeline()->pushModule(assignMaterial);


	Modeling::initStaticPlugin();
	RigidBody::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	HeightFieldLibrary::initStaticPlugin();
	DualParticleSystem::initStaticPlugin();
	Peridynamics::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();
	Volume::initStaticPlugin();
	Multiphysics::initStaticPlugin();
	dynoIO::initStaticPlugin();

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1920, 1080);
	app.mainLoop();

	return 0;
}