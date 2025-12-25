#include <UbiApp.h>
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
#include "GLRenderEngine.h"
#include "RenderWindow.h"
#include "BasicShapes/PlaneModel.h"
#include "TextureMeshLoader.h"
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

	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	MaterialManager::createCustomMaterial();
	
	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(std::string(getAssetPath() + "Jeep/JeepGltf/jeep.gltf"));

	auto gltfWireframe = gltf->graphicsPipeline()->findFirstModule<GLWireframeVisualModule>();
	if (gltfWireframe)
	{
		gltf->graphicsPipeline()->popModule(gltfWireframe);
	}

	auto srcMaterial = MaterialManager::getMaterial("Body1");
	std::shared_ptr<CustomMaterial> customMaterial = NULL;
	if (srcMaterial) 
	{
		customMaterial = MaterialManager::createCustomMaterial(srcMaterial);
		customMaterial->varEmissiveIntensity()->setValue(1.0f);
		auto matPipeline = customMaterial->materialPipeline();

		auto texCorrect = std::make_shared<ColorCorrect>();
		srcMaterial->outTexColor()->connect(texCorrect->inTexture());
		texCorrect->varSaturation()->setValue(1.158f);
		texCorrect->varHUEOffset()->setValue(206.557f);
		texCorrect->varContrast()->setValue(1.021f);
		texCorrect->varGamma()->setValue(1.936f);
		texCorrect->outTexture()->connect(customMaterial->inTexColor());
		matPipeline->pushModule(texCorrect);

		auto breakTex = std::make_shared<BreakTexture>();
		srcMaterial->outTexColor()->connect(breakTex->inTexture());
		auto makeTex = std::make_shared<MakeTexture>();
		breakTex->outR()->connect(makeTex->inR());
		breakTex->outG()->connect(makeTex->inG());
		breakTex->outB()->connect(makeTex->inB());
		breakTex->outA()->connect(makeTex->inA());
		makeTex->outTexture()->connect(customMaterial->inBaseColor());
		matPipeline->pushModule(breakTex);
		matPipeline->pushModule(makeTex);


		auto image = std::make_shared<ImageLoaderModule>();
		image->varImagePath()->setValue(std::string(getAssetPath() + "Jeep/JeepGltf/jeep_body_camouflage.png"));
		matPipeline->pushModule(image);

		matPipeline->updateMaterialPipline();

		auto assignMaterial = std::make_shared<AssignTextureMeshMaterial<DataType3f>>();
		assignMaterial->varShapeIndex()->setValue(5);
		assignMaterial->varMaterialName()->setValue(customMaterial->getName());
		auto textureRender = gltf->graphicsPipeline()->findFirstModule<GLPhotorealisticRender>();
		if (textureRender)
			assignMaterial->outTextureMesh()->connect(textureRender->inTextureMesh());
		gltf->stateTextureMesh()->connect(assignMaterial->inTextureMesh());
		gltf->graphicsPipeline()->pushModule(assignMaterial);


		auto emissiveInput = std::make_shared<MatInput>();
		matPipeline->pushModule(emissiveInput);
		srcMaterial->outEmissiveItensity()->disconnect(customMaterial->inEmissiveIntensity());
		emissiveInput->outValue()->connect(customMaterial->inEmissiveIntensity());

		//auto emissiveCorrect = std::make_shared<ColorCorrect>();
		//srcMaterial->outTexEmissive()->connect(emissiveCorrect->inTexture());
		//emissiveInput->outValue()->connect(emissiveCorrect->inBrightness());
		//emissiveCorrect->outTexture()->connect(customMaterial->inTexEmissiveColor());
		//matPipeline->pushModule(emissiveCorrect);
	}

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(5.0,0.0,5.0));
	auto planeWireframe = plane->graphicsPipeline()->findFirstModule<GLWireframeVisualModule>();
	if (planeWireframe)
	{
		plane->graphicsPipeline()->popModule(planeWireframe);
	}




	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(scn);
	app.initialize(1920, 1080);

	// setup envmap
	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->setUseEnvmapBackground(false);
		renderer->setEnvmapScale(3.0f);
		renderer->showGround = false;
			
	}

	app.renderWindow()->setShadowMultiplier(1.0f);
	app.renderWindow()->setShadowBrightness(0.14f);
	app.renderWindow()->setSamplePower(3.27f);
	app.renderWindow()->setShadowContrast(3.90f);

	app.mainLoop();

	return 0;
}