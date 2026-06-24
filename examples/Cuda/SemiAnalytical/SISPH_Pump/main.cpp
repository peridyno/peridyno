#include "QtGUI/QtApp.h"

#include "SceneGraph.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

#include "initializeModeling.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "ObjIO/ObjLoader.h"
#include <SemiAnalyticalScheme/SemiAnalyticalParticleFluid.h>
#include <GLRenderEngine.h>
#include <Commands/Merge.h>
#include <ParticleSystem/Emitters/CircularEmitter.h>
#include <ParticleSystem/Viscosity/ImplicitViscosity.h>
#include <SemiAnalyticalScheme/Module/SemiAnalyticalDensitySolver.h>
#include <ParticleSystem/Module/SurfaceEnergyForce.h>
#include <Module/CalculateNorm.h>


using namespace std;
using namespace dyno;
std::shared_ptr<SceneGraph> createScene()
{
	Vec3f gravity = Vec3f(0.0f, -9.8f, 0.0f);
	gravity *= 0.1;
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(gravity);
	scn->setLowerBound(Vec3f(-4.0f, -4.0f, -4.0f));
	scn->setUpperBound(Vec3f(4.0f, 4.0f, 4.0f));
	//Create a particle emitter
	auto emitter = scn->addNode(std::make_shared<CircularEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(-0.730f, 1.29f, 0.0f));
	emitter->varRotation()->setValue(Vec3f(0, 0, 90));
	emitter->varScale()->setValue(Vec3f(0.8, 0.8, 0.8));
	emitter->varVelocityMagnitude()->setValue(1.5f);

	auto sRenderf = std::make_shared<GLSurfaceVisualModule>();
	//sRenderf->varBaseColor()->setValue(Color(0.8f, 0.52f, 0.25f));
	sRenderf->varBaseColor()->setValue(Color(1.0f, 1.0f, 1.0f));
	sRenderf->varAlpha()->setValue(0.2f);
	sRenderf->varVisible()->setValue(true);

	auto sRenderf2 = std::make_shared<GLSurfaceVisualModule>();
	sRenderf2->varBaseColor()->setValue(Color(1.0, 1.0, 1.0));
	sRenderf2->varAlpha()->setValue(1.0f);
	sRenderf2->varMetallic()->setValue(1.0);
	sRenderf2->varRoughness()->setValue(0.25);
	sRenderf2->varAlpha()->setValue(1.0);
	sRenderf2->varVisible()->setValue(true);
	auto sRenderf3 = std::make_shared<GLSurfaceVisualModule>();
	sRenderf3->varBaseColor()->setValue(Color(1.0, 1.0, 1.0));
	sRenderf3->varAlpha()->setValue(1.0f);
	sRenderf3->varMetallic()->setValue(1.0);
	sRenderf3->varRoughness()->setValue(0.25);
	sRenderf3->varAlpha()->setValue(1.0);
	sRenderf3->varVisible()->setValue(true);
	auto sRenderf4 = std::make_shared<GLSurfaceVisualModule>();
	sRenderf4->varBaseColor()->setValue(Color(0.8f, 0.52f, 0.25f));
	sRenderf4->varAlpha()->setValue(1.0f);
	sRenderf4->varMetallic()->setValue(1.0);
	sRenderf4->varRoughness()->setValue(0.25);
	sRenderf4->varAlpha()->setValue(0.1);
	sRenderf4->varVisible()->setValue(true);
	auto sRenderf5 = std::make_shared<GLSurfaceVisualModule>();
	sRenderf5->varBaseColor()->setValue(Color(0.8f, 0.52f, 0.25f));
	sRenderf5->varAlpha()->setValue(1.0f);
	sRenderf5->varMetallic()->setValue(1.0);
	sRenderf5->varRoughness()->setValue(0.25);
	sRenderf5->varAlpha()->setValue(0.1);
	sRenderf5->varVisible()->setValue(true);
	auto sRenderf6 = std::make_shared<GLSurfaceVisualModule>();
	sRenderf6->varBaseColor()->setValue(Color(1.0, 1.0, 1.0));
	sRenderf6->varAlpha()->setValue(1.0f);
	sRenderf6->varMetallic()->setValue(1.0);
	sRenderf6->varRoughness()->setValue(0.25);
	sRenderf6->varAlpha()->setValue(0.1);
	sRenderf6->varVisible()->setValue(true);

	auto objLoader = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	objLoader->graphicsPipeline()->clear();
	objLoader->varFileName()->setValue(getAssetPath() + "obj/pump/Inner.obj");
	objLoader->varLocation()->setValue(Vec3f(0, 1.575, 0));
	objLoader->varScale()->setValue(Vec3f(0.302, 0.302, 0.350));
	objLoader->varCenter()->setValue(Vec3f(0, 1.575, 0.0));
	objLoader->varVelocity()->setValue(Vec3f(0.0f, 0.0f, 0.0f));
	//objLoader->varRotation()->setValue(Vec3f(0, 90, 0));
	objLoader->outTriangleSet()->connect(sRenderf3->inTriangleSet());
	objLoader->graphicsPipeline()->pushModule(sRenderf3);
	objLoader->setForceUpdate(true);

	auto objLoader2 = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	objLoader2->graphicsPipeline()->clear();
	objLoader2->varFileName()->setValue(getAssetPath() + "obj/pump/Inner.obj");
	objLoader2->varLocation()->setValue(Vec3f(0, 1.0, 0));
	objLoader2->varScale()->setValue(Vec3f(0.302, 0.302, 0.350));
	objLoader2->varCenter()->setValue(Vec3f(0, 1.0, 0));
	objLoader2->varVelocity()->setValue(Vec3f(0.0f, 0.0f, 0.0f));
	//objLoader2->varRotation()->setValue(Vec3f(0, 90, 0));
	objLoader2->outTriangleSet()->connect(sRenderf2->inTriangleSet());
	objLoader2->graphicsPipeline()->pushModule(sRenderf2);
	objLoader2->setForceUpdate(true);


	auto shaft1 = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	shaft1->graphicsPipeline()->clear();
	shaft1->varFileName()->setValue(getAssetPath() + "obj/pump/Inner_2.obj");
	shaft1->varLocation()->setValue(Vec3f(0, 1.575, 0));
	shaft1->varScale()->setValue(Vec3f(0.302, 0.302, 0.350));
	shaft1->varCenter()->setValue(Vec3f(0, 1.575, 0.0));
	shaft1->varVelocity()->setValue(Vec3f(0.0f, 0.0f, 0.0f));
	shaft1->outTriangleSet()->connect(sRenderf4->inTriangleSet());
	shaft1->graphicsPipeline()->pushModule(sRenderf4);
	shaft1->setForceUpdate(true);
	shaft1->graphicsPipeline()->disable();

	auto shaft2 = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	shaft2->graphicsPipeline()->clear();
	shaft2->varFileName()->setValue(getAssetPath() + "obj/pump/Inner_2.obj");
	shaft2->varLocation()->setValue(Vec3f(0, 1.0, 0));
	shaft2->varScale()->setValue(Vec3f(0.302, 0.302, 0.350));
	shaft2->varCenter()->setValue(Vec3f(0, 1.0, 0));
	shaft2->varVelocity()->setValue(Vec3f(0.0f, 0.0f, 0.0f));
	shaft2->outTriangleSet()->connect(sRenderf5->inTriangleSet());
	shaft2->graphicsPipeline()->pushModule(sRenderf5);
	shaft2->setForceUpdate(true);
	shaft2->graphicsPipeline()->disable();

	auto barricade = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	barricade->varFileName()->setValue(getAssetPath() + "obj/pump/ClipMesh.obj");
	//barricade->varFileName()->setValue(getAssetPath() + "obj/SimMesh2.obj");
	barricade->varLocation()->setValue(Vec3f(0, 1.0, 0));
	//barricade->varRotation()->setValue(Vec3f(0, 180, 0));
	barricade->varScale()->setValue(Vec3f(0.3, 0.3, 0.3));
	barricade->graphicsPipeline()->clear();
	barricade->outTriangleSet()->connect(sRenderf6->inTriangleSet());
	barricade->graphicsPipeline()->pushModule(sRenderf6);

	auto background = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	background->varFileName()->setValue(getAssetPath() + "obj/pump/Background.obj");
	background->varLocation()->setValue(Vec3f(0, 1.0, 0));
	background->varScale()->setValue(Vec3f(0.3, 0.3, 0.3));
	background->graphicsPipeline()->disable();

	auto cap = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	cap->varFileName()->setValue(getAssetPath() + "obj/pump/Cap.obj");
	cap->varLocation()->setValue(Vec3f(0, 1.0, 0));
	cap->varScale()->setValue(Vec3f(0.3, 0.3, 0.302));
	cap->graphicsPipeline()->clear();
	cap->outTriangleSet()->connect(sRenderf->inTriangleSet());
	//cap->graphicsPipeline()->pushModule(sRenderf);
	cap->graphicsPipeline()->disable();

	auto merge = scn->addNode(std::make_shared<Merge<DataType3f>>());
	merge->varUpdateMode()->setCurrentKey(1);
	objLoader->outTriangleSet()->connect(merge->inTriangleSets());
	objLoader2->outTriangleSet()->connect(merge->inTriangleSets());
	barricade->outTriangleSet()->connect(merge->inTriangleSets());
	cap->outTriangleSet()->connect(merge->inTriangleSets());
	merge->graphicsPipeline()->clear();

#ifdef LARGETIME
	Real dt = 0.005;
#else
	Real dt = 0.001;
#endif // LARGETIME

	//SFI node
	Real scale = 1.0f;
	scale *= 1.5;
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalParticleFluid<DataType3f>>());
	objLoader->varAngularVelocity()->setValue(Vec3f(0.0f, 0.0f, -1.00) * scale);
	objLoader2->varAngularVelocity()->setValue(Vec3f(0.0f, 0.0f, 1.00) * scale);
	shaft1->varAngularVelocity()->setValue(Vec3f(0.0f, 0.0f, -1.00) * scale);
	shaft2->varAngularVelocity()->setValue(Vec3f(0.0f, 0.0f, 1.00) * scale);

	sfi->setDt(dt);
	sfi->varSamplingDistance()->setValue(0.005f);
	sfi->varSmoothingLength()->setValue(1.5);
	sfi->varSearchRadius()->setValue(sfi->varSmoothingLength()->getValue() * sfi->varSamplingDistance()->getValue() * 2.01);
	{
		auto solver = sfi->animationPipeline()->findFirstModule<SemiAnalyticalDensitySolver<DataType3f>>();
		solver->varIterationNumber()->setValue(5);
		solver->varMu()->setValue(1.0);
		solver->varKappaLower()->setValue(2000.0f);
		solver->varWarmStart()->setValue(true);
		solver->varPolynomialNumber()->setValue(3);
		solver->varD_hat()->setValue(sfi->varSamplingDistance()->getValue());

		auto surfacesolver = sfi->animationPipeline()->findFirstModule<SurfaceEnergyForce<DataType3f>>();
		surfacesolver->varKappa()->setValue(0.000);
		auto viscositysolver = sfi->animationPipeline()->findFirstModule<ImplicitViscosity<DataType3f>>();
		viscositysolver->varViscosity()->setValue(5);
	}

	emitter->connect(sfi->importParticleEmitters());
	{
		sfi->graphicsPipeline()->clear();
		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->varPointSize()->setValue(0.004);
		ptRender->varBaseColor()->setValue(Color(1, 0, 0));
		ptRender->varMetallic()->setValue(1.0f);
		ptRender->varRoughness()->setValue(1.0f);
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(3.0f);
		sfi->stateVelocity()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());
		sfi->graphicsPipeline()->pushModule(calculateNorm);
		colorMapper->outColor()->connect(ptRender->inColor());
		sfi->statePointSet()->connect(ptRender->inPointSet());
		sfi->graphicsPipeline()->pushModule(colorMapper);
		sfi->graphicsPipeline()->pushModule(ptRender);
	}

	merge->stateTriangleSets()->connect(sfi->inTriangleSets());


	scn->printNodeInfo(true);
	scn->printSimulationInfo(true);
	return scn;
}
int main()
{
	Modeling::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();

	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.renderWindow()->getCamera()->setEyePos(Vec3f(0.0f, 1.4f, 2.16f));//global view
	app.renderWindow()->getCamera()->setTargetPos(Vec3f(0.00, 1.29, -0.53));//global view
	app.renderWindow()->setMainLightDirection(glm::vec3(-0.3f, -0.3f, -0.9f));
	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setUseEnvmapBackground(false);

		renderer->bgColor0 = { 1, 1, 1 };
		renderer->bgColor1 = { 1, 1, 1 };

		renderer->planeColor = { 1,1,1,1 };
		renderer->rulerColor = { 1,1,1,1 };
		renderer->showGround = false;
		auto& light = app.renderWindow()->getRenderParams().light;
		light.mainLightScale = 10;
	}
	app.mainLoop();

	return 0;
}
