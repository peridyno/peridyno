#include "QtGUI/QtApp.h"

#include "SceneGraph.h"

#include "SemiAnalyticalScheme/SemiAnalyticalParticleFluid.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include "initializeModeling.h"
#include "ParticleSystem/initializeParticleSystem.h"

#include "ObjIO/ObjLoader.h"
#include <BasicShapes/CubeModel.h>
#include <Commands/Merge.h>
#include <GLRenderEngine.h>
#include <Samplers/ShapeSampler.h>
#include <SemiAnalyticalScheme/Module/SemiAnalyticalDensitySolver.h>

#include <SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h>
#include <ParticleSystem/Module/SurfaceEnergyForce.h>
#include <ParticleSystem/Module/ImplicitViscosity.h>
#include <ParticleSystem/Emitters/SquareEmitter.h>



using namespace std;
using namespace dyno;
std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	Real dt = 0.001;
	Real SAMPLINGDISTANCE = 0.005;

	auto emitter = scn->addNode(std::make_shared<SquareEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(0, 1.470, 0));
	emitter->varScale()->setValue(Vec3f(1.2, 1.2, 1.2));
	emitter->varVelocityMagnitude()->setValue(1.0f);
	emitter->varBeginFrame()->setValue(0);
	emitter->varStopFrame()->setValue(250);
	emitter->graphicsPipeline()->clear();

	auto objLoader = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	objLoader->setDt(dt);
	objLoader->graphicsPipeline()->clear();
	objLoader->varFileName()->setValue(getAssetPath() + "obj/ClosedtriCone.obj");
	//objLoader->varBeginFrame()->setValue(800);
	objLoader->varLocation()->setValue(Vec3f(0, 0.3, 0));
	objLoader->varScale()->setValue(Vec3f(0.04, 0.3, 0.04));
	objLoader->varRotation()->setValue(Vec3f(0, 0, 45));
	objLoader->varCenter()->setValue(Vec3f(0.0f, 0.9f, 0.0f));
	objLoader->varAngularVelocity()->setValue(Vec3f(0.0f, 0.0f, 0.0f));
	//objLoader->varBufferFrame()->setValue(300);
	objLoader->graphicsPipeline()->clear();
	auto merge = scn->addNode(std::make_shared<Merge<DataType3f>>());
	merge->setDt(dt);
	merge->varUpdateMode()->setCurrentKey(1);
	objLoader->outTriangleSet()->connect(merge->inTriangleSets());
	merge->graphicsPipeline()->clear();

	auto wireframe = std::make_shared<GLWireframeVisualModule>();
	merge->stateTriangleSets()->connect(wireframe->inEdgeSet());
	merge->graphicsPipeline()->pushModule(wireframe);


	//SFI node
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalParticleFluid<DataType3f>>());
	sfi->setDt(dt);
	sfi->varSamplingDistance()->setValue(SAMPLINGDISTANCE);
	merge->stateTriangleSets()->connect(sfi->inTriangleSets());
	sfi->varSmoothingLength()->setValue(1.5);
	auto solver = sfi->animationPipeline()->findFirstModule<SemiAnalyticalDensitySolver<DataType3f>>();
	solver->varIterationNumber()->setValue(5);
	solver->varMu()->setValue(1.0);
	solver->varBoundaryFriction()->setValue(0.0001f);
	solver->varKappaLower()->setValue(100.0f);
	solver->varPolynomialNumber()->setValue(5);
	solver->varD_hat()->setValue(sfi->varSamplingDistance()->getValue() * 2);

	auto viscositysolver = sfi->animationPipeline()->findFirstModule<ImplicitViscosity<DataType3f>>();
	viscositysolver->varViscosity()->setValue(5.0);
	auto surfacesolver = sfi->animationPipeline()->findFirstModule<SurfaceEnergyForce<DataType3f>>();
	surfacesolver->varKappa()->setValue(0.00);
	//initialParticles->connect(sfi->importInitialStates());
	emitter->connect(sfi->importParticleEmitters());
	{
		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->varPointSize()->setValue(0.005);
		ptRender->varRoughness()->setValue(1.0);
		ptRender->varBaseColor()->setValue(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();

		colorMapper->varMax()->setValue(3.0f);
		sfi->stateVelocity()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());

		colorMapper->outColor()->connect(ptRender->inColor());
		sfi->statePointSet()->connect(ptRender->inPointSet());

		sfi->graphicsPipeline()->clear();
		sfi->graphicsPipeline()->pushModule(calculateNorm);
		sfi->graphicsPipeline()->pushModule(colorMapper);
		sfi->graphicsPipeline()->pushModule(ptRender);
	}
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

	app.renderWindow()->getCamera()->setEyePos(Vec3f(0.0f, 1.12f, 1.73f));
	app.renderWindow()->getCamera()->setTargetPos(Vec3f(0.00f, 0.87f, -0.32f));
	app.renderWindow()->setMainLightDirection(glm::vec3(0.00f, 0.00f, -1.0f));

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
		light.mainLightShadow = 0;
	}
	//Log::setUserReceiver(&RecieveLogMessage);
	app.mainLoop();


	return 0;
}