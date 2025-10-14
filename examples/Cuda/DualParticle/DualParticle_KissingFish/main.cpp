#include <UbiApp.h>
#include <SceneGraph.h>

///Points Sampler
#include <PointsLoader.h>
#include <BasicShapes/SphereModel.h>

///Boundary
#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>
#include <StaticMeshLoader.h>
#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>

///Fluid Solver
#include "DualParticleSystem/Module/DualParticleIsphModule.h"
#include <DualParticleSystem/Module/ThinFeature.h>
#include <DualParticleSystem/Module/VirtualFissionFusionStrategy.h>
#include <DualParticleSystem/DualParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <ParticleSystem/Emitters/PoissonEmitter.h>

///Renderer
#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(3.0, 3, 3.0));
	scn->setLowerBound(Vec3f(-3.0, -3.0, -3.0));
	scn->setGravity(Vec3f(0.0f, 0.0f, -3.8f));

	auto ptsLoader = scn->addNode(std::make_shared<PointsLoader<DataType3f>>());
	ptsLoader->varFileName()->setValue(getAssetPath() + "fish/BigFishPoints.obj");
	
	ptsLoader->varRotation()->setValue(Vec3f(0.0f, 0.0, 3.1415926f));
	ptsLoader->varLocation()->setValue(Vec3f(0.0f, 0.6f, 0.30f));
	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f >>());
	ptsLoader->outPointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<DualParticleFluid<DataType3f>>(
		DualParticleFluid<DataType3f>::FissionFusionStrategy));

	initialParticles->connect(fluid->importInitialStates());
	fluid->graphicsPipeline()->clear();

	auto fish = scn->addNode(std::make_shared<StaticMeshLoader<DataType3f>>());
	fish->varFileName()->setValue(getAssetPath() + "fish/FishMesh.obj");
	fish->varScale()->setValue(Vec3f(1.5));
	fish->varLocation()->setValue(Vec3f(0.0, 0.6, -0.4));
	fish->graphicsPipeline()->clear();
	auto sRenderf = std::make_shared<GLSurfaceVisualModule>();
	sRenderf->setColor(Color(0.8f, 0.52f, 0.25f));
	sRenderf->setVisible(true);
	sRenderf->varUseVertexNormal()->setValue(true);	// use generated smooth normal
	sRenderf->varMetallic()->setValue(1.0);
	fish->stateTriangleSet()->connect(sRenderf->inTriangleSet());
	fish->graphicsPipeline()->pushModule(sRenderf);

	auto pm_collide = scn->addNode(std::make_shared <TriangularMeshBoundary<DataType3f >>());
	fish->stateTriangleSet()->connect(pm_collide->inTriangleSet());
	fluid->connect(pm_collide->importParticleSystems());
	
	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->varPointSize()->setValue(0.004);
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

	fluid->statePointSet()->connect(ptRender->inPointSet());
	colorMapper->outColor()->connect(ptRender->inColor());
	fluid->graphicsPipeline()->pushModule(ptRender);

	// A simple color bar widget for node
	//auto colorBar = std::make_shared<ImColorbar>();
	//colorBar->varMax()->setValue(1.0f);
	//colorBar->varFieldName()->setValue("Velocity");
	//calculateNorm->outNorm()->connect(colorBar->inScalar());
	//// add the widget to app
	//fluid->graphicsPipeline()->pushModule(colorBar);

	//auto vpRender = std::make_shared<GLPointVisualModule>();
	//vpRender->setColor(Color(1, 1, 0));
	//vpRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	//fluid->stateVirtualPointSet()->connect(vpRender->inPointSet());
	//vpRender->varPointSize()->setValue(0.0005);
	//fluid->graphicsPipeline()->pushModule(vpRender);

	return scn;
}

int main()
{

	UbiApp window(GUIType::GUI_GLFW);
	window.initialize(2048, 1080);
	window.setSceneGraph(createScene());

	window.setSceneGraph(createScene());
	window.initialize(1366, 768);
	auto cam = window.renderWindow()->getCamera();

	cam->setEyePos(Vec3f(1.71378, 1.24788, 0.404752));
	cam->setTargetPos(Vec3f(-0.172568, 0.750952, -0.466298));


	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(window.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->showGround = false;
		renderer->setUseEnvmapBackground(false);
	}

	window.mainLoop();

	return 0;
}


