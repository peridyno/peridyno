#include <UbiApp.h>
using namespace dyno;

#include "RigidBody/initializeRigidBody.h"
#include "ParticleSystem/initializeParticleSystem.h"
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
#include "GLCarInstanceRender.h"
#include "RigidBody/Vehicle.h"
#include "RigidBody/MultibodySystem.h"
#include "CarMaterial.h"
#include "LightController.h"
#include "MaterialEditModule.h"
#include "RigidBody/Module/InstanceTransform.h"
/**
 * @brief This is a sample demonstrating a custom material. 
	1.Run the simulation
	2.press the Q key to turn the car headlights on/off.
	3.press the A/D keys to activate the turn signals.
	4.press the S/W keys to turn the reverse lights on/off.
 */

#define USEMULTIBODY
int main()
{

	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	auto jeep = scn->addNode(std::make_shared<Jeep<DataType3f>>());

	jeep->reset();
	std::vector<Transform3f> temp;
	temp.push_back(Transform3f());
	temp.push_back(Transform3f(Vec3f(-5,0,0),Mat3f::identityMatrix()));
	jeep->varVehiclesTransform()->setValue(temp);
	auto gltfWireframe = jeep->graphicsPipeline()->findFirstModule<GLWireframeVisualModule>();
	if (gltfWireframe)
	{
		jeep->graphicsPipeline()->popModule(gltfWireframe);
	}

#ifdef USEMULTIBODY

	auto multiBodySystem = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	jeep->connect(multiBodySystem->importVehicles());

#endif // USEMULTIBODY

	auto srcMaterial = std::dynamic_pointer_cast<MaterialLoaderModule>(MaterialManager::getMaterialManagedModule("Body1"));

	if (srcMaterial) 
	{
		//Create Material
		std::shared_ptr<CustomCarMaterial> carMat;
		{
			std::shared_ptr<BreakMaterial> breakCarMat;
			carMat = std::make_shared<CustomCarMaterial>(srcMaterial, breakCarMat, std::string("JeepBodyMat"));
			MaterialManager::addCustomMaterial(carMat);
			auto carMatPipeline = carMat->materialPipeline();

			auto headLightMask = std::make_shared<ImageLoaderModule>();
			headLightMask->varImagePath()->setValue(std::string(getAssetPath() + "Jeep/JeepGltf/HeadLight.png"));

			auto brakeLight = std::make_shared<ImageLoaderModule>();
			brakeLight->varImagePath()->setValue(std::string(getAssetPath() + "Jeep/JeepGltf/BrakeLight.png"));

			
			auto makeLightTex = std::make_shared<MakeTexture>();

			headLightMask->outGrayImage()->connect(makeLightTex->inR());
			brakeLight->outGrayImage()->connect(makeLightTex->inG());
			brakeLight->outGrayImage()->connect(makeLightTex->inB());

			makeLightTex->outTexture()->connect(carMat->inTexLightMask());

			carMatPipeline->pushModule(headLightMask);
			carMatPipeline->pushModule(brakeLight);
			carMatPipeline->pushModule(makeLightTex);

			auto texCorrect = std::make_shared<ColorCorrect>();
			breakCarMat->outTexColor()->connect(texCorrect->inTexture());
			texCorrect->varSaturation()->setValue(1.158f);
			texCorrect->varHUEOffset()->setValue(206.557f);
			texCorrect->varContrast()->setValue(1.021f);
			texCorrect->varGamma()->setValue(1.936f);
			texCorrect->outTexture()->connect(carMat->inTexColor());
			carMatPipeline->pushModule(texCorrect);

			carMatPipeline->updateMaterialPipline();
		}

		//AssignMaterial
		auto assignMaterial = std::make_shared<AssignTextureMeshMaterial<DataType3f>>();
		assignMaterial->varShapeIndex()->setValue(5);
		assignMaterial->varMaterialName()->setValue(carMat->getName());
		auto textureRender = jeep->graphicsPipeline()->findFirstModule<GLPhotorealisticRender>();
		if (textureRender)
			assignMaterial->outTextureMesh()->connect(textureRender->inTextureMesh());
		jeep->stateTextureMesh()->connect(assignMaterial->inTextureMesh());
		jeep->graphicsPipeline()->pushModule(assignMaterial);

		//Create CarRender
		{
			auto instance = jeep->graphicsPipeline()->findFirstModule<InstanceTransform<DataType3f>>();

			auto carRender = std::make_shared<GLCarInstanceRender>();
			assignMaterial->outTextureMesh()->connect(carRender->inTextureMesh());
			instance->outInstanceTransform()->connect(carRender->inTransform());
			jeep->graphicsPipeline()->pushModule(carRender);
			

			auto lightController = std::make_shared<LightController>();
			jeep->stateInstanceArticulatedBodyID()->connect(lightController->inShapeVehicleID());
			lightController->outHeadLight()->connect(carRender->inHeadLight());
			lightController->outTurnSignal()->connect(carRender->inTurnSignal());
			lightController->outBrakeLight()->connect(carRender->inBrakeLight());
			lightController->outLightDirection()->connect(carRender->inRightDirection());

			jeep->graphicsPipeline()->pushModule(lightController);

			auto photo = jeep->graphicsPipeline()->findFirstModule<GLPhotorealisticInstanceRender>();
			jeep->graphicsPipeline()->popModule(photo);

		}	
	}

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(25.0,0.0,125.0));
	auto planeWireframe = plane->graphicsPipeline()->findFirstModule<GLWireframeVisualModule>();
	if (planeWireframe)
	{
		plane->graphicsPipeline()->popModule(planeWireframe);
	}

#ifdef USEMULTIBODY
	plane->stateTriangleSet()->connect(multiBodySystem->inTriangleSet());
#endif // USEMULTIBODY

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
		renderer->forceRender = true;
		
	}

	app.renderWindow()->setShadowMultiplier(1.0f);
	app.renderWindow()->setShadowBrightness(0.14f);
	app.renderWindow()->setSamplePower(3.27f);
	app.renderWindow()->setShadowContrast(3.90f);

	app.mainLoop();

	return 0;
}