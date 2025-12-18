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
#include <BasicShapes/CubeModel.h>
#include "GltfLoader.h"
#include "MaterialEditModule.h"
#include "ImageLoader.h"
#include "GLPhotorealisticInstanceRender.h"

/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(1.5, 1, 1.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));

	auto cube1 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	//auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());

	return scn;
}

int main()
{
	MaterialManager::NewMaterial();
	MaterialManager::createCustomMaterial();

	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf"));

	auto srcMaterial = MaterialManager::getMaterial("BodyMaterial");
	std::shared_ptr<CustomMaterial> customMaterial = NULL;
	if (srcMaterial) 
	{
		customMaterial = MaterialManager::createCustomMaterial(srcMaterial);
		gltf->graphicsPipeline()->pushModule(srcMaterial);

		auto matPipeline = customMaterial->materialPipeline();
		gltf->graphicsPipeline()->pushModule(customMaterial);

		auto texCorrect = std::make_shared<ColorCorrect>();
		srcMaterial->outTexColor()->connect(texCorrect->inTexture());
		texCorrect->varSaturation()->setValue(0);
		texCorrect->outTexture()->connect(customMaterial->inTexColor());
		matPipeline->pushModule(texCorrect);
		gltf->graphicsPipeline()->pushModule(texCorrect);

		auto image = std::make_shared<ImageLoaderModule>();
		image->varImagePath()->setValue(std::string(getAssetPath() + "Jeep/JeepGltf/jeep_body_camouflage.png"));
		matPipeline->pushModule(image);

		matPipeline->updateMaterialPipline();
		gltf->graphicsPipeline()->pushModule(image);

		auto assignMaterial = std::make_shared<AssignTextureMeshMaterial<DataType3f>>();
		assignMaterial->varShapeIndex()->setValue(5);
		assignMaterial->varMaterialName()->setValue(customMaterial->getName());
		auto textureRender = gltf->graphicsPipeline()->findFirstModule<GLPhotorealisticRender>();
		if (textureRender)
			assignMaterial->outTextureMesh()->connect(textureRender->inTextureMesh());
		gltf->stateTextureMesh()->connect(assignMaterial->inTextureMesh());
		gltf->graphicsPipeline()->pushModule(assignMaterial);
	}


	


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