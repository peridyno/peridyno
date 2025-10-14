#include <UbiApp.h>
#include <SceneGraph.h>

///Particle fluid solver
#include <DualParticleSystem/DualParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>

///Particle Sampler
#include <PointsLoader.h>

///Render
#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>

using namespace std;
using namespace dyno;

bool useVTK = false;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(3.0, 3.0, 3.0));
	scn->setLowerBound(Vec3f(-3.0, -3.0, -3.0));
	scn->setGravity(Vec3f(0.0f));

	auto ptsLoader = scn->addNode(std::make_shared<PointsLoader<DataType3f>>());
	ptsLoader->varFileName()->setValue(getAssetPath() + "fish/FishPoints.obj");
	ptsLoader->varRotation()->setValue(Vec3f(0.0f, 0.0f, 3.1415926f));
	ptsLoader->varLocation()->setValue(Vec3f(0.0f, 0.0f, 0.23f));
	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f >>());
	initialParticles->varInitialVelocity()->setValue(Vec3f(0.0f, 0.0f, -1.5f));
	ptsLoader->outPointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto ptsLoader2 = scn->addNode(std::make_shared<PointsLoader<DataType3f>>());
	ptsLoader2->varFileName()->setValue(getAssetPath() + "fish/FishPoints.obj");
	ptsLoader2->varRotation()->setValue(Vec3f(0.0f, 0.0f, 0.0));
	ptsLoader2->varLocation()->setValue(Vec3f(0.0f, 0.0f, -0.23f));
	auto initialParticles2 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f >>());
	initialParticles2->varInitialVelocity()->setValue(Vec3f(0.0f, 0.0f, 1.5f));
	ptsLoader2->outPointSet()->promoteOuput()->connect(initialParticles2->inPoints());

	auto fluid = scn->addNode(std::make_shared<DualParticleFluid<DataType3f>>(DualParticleFluid<DataType3f>::FissionFusionStrategy));
	fluid->varReshuffleParticles()->setValue(true);
	initialParticles->connect(fluid->importInitialStates());
	initialParticles2->connect(fluid->importInitialStates());

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->varPointSize()->setValue(0.0025f);
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


	auto vpRender = std::make_shared<GLPointVisualModule>();
	vpRender->setColor(Color(1, 1, 0));
	vpRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	fluid->stateVirtualPointSet()->connect(vpRender->inPointSet());
	vpRender->varPointSize()->setValue(0.0005);
	fluid->graphicsPipeline()->pushModule(vpRender);

	return scn;
}

int main()
{
	UbiApp window;
	window.setSceneGraph(createScene());
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}


