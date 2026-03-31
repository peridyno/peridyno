#include <QtApp.h>
#include <SceneGraph.h>

///Particle fluid solver
#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <ParticleSystem/Emitters/PoissonEmitter.h>

///Particle Sampling
#include <PointsLoader.h>

///Mesh boundary
#include <BasicShapes/SphereModel.h>
#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>

///Renderer
#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(3.0, 3, 3.0));
	scn->setLowerBound(Vec3f(-3.0, -3.0, -3.0));

	auto ptsLoader = scn->addNode(std::make_shared<PointsLoader<DataType3f>>());
	ptsLoader->varFileName()->setValue(getAssetPath() + "fish/FishPoints.obj");
	ptsLoader->varRotation()->setValue(Vec3f(0.0f, 3.14 * 2 / 5, 0.0f));
	ptsLoader->varLocation()->setValue(Vec3f(0.0f, 0.4f, 0.30f));
	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f >>());
	ptsLoader->outPointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->varIncompressibilitySolver()->getDataPtr()->setCurrentKey(ParticleFluid<DataType3f>::DualParticle);
	fluid->setDt(0.001);
	fluid->varSmoothingLength()->setValue(2.4);
	initialParticles->connect(fluid->importInitialStates());

	auto ball = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	ball->varScale()->setValue(Vec3f(0.38));
	ball->varLocation()->setValue(Vec3f(0.0, 0.0, 0.3));
	auto sRenderf = std::make_shared<GLSurfaceVisualModule>();
	sRenderf->varBaseColor()->setValue(Color(0.8f, 0.52f, 0.25f));
	sRenderf->varVisible()->setValue(true);
	sRenderf->varUseVertexNormal()->setValue(true);	// use generated smooth normal
	ball->stateTriangleSet()->connect(sRenderf->inTriangleSet());
	ball->graphicsPipeline()->pushModule(sRenderf);

	auto pm_collide = scn->addNode(std::make_shared <TriangularMeshBoundary<DataType3f >>());
	ball->stateTriangleSet()->connect(pm_collide->inTriangleSet());
	fluid->connect(pm_collide->importParticleSystems());
	
	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->varBaseColor()->setValue(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

	fluid->statePointSet()->connect(ptRender->inPointSet());
	colorMapper->outColor()->connect(ptRender->inColor());
	fluid->graphicsPipeline()->pushModule(ptRender);

	// A simple color bar widget for node
	auto colorBar = std::make_shared<ImColorbar>();
	colorBar->varMax()->setValue(5.0f);
	colorBar->varFieldName()->setValue("Velocity");
	calculateNorm->outNorm()->connect(colorBar->inScalar());
	// add the widget to app
	fluid->graphicsPipeline()->pushModule(colorBar);

	return scn;
}

int main()
{
	QtApp window;
	window.setSceneGraph(createScene());
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}


