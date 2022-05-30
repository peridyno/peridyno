#include <QtApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/ParticleEmitterSquare.h>

#include <Module/CalculateNorm.h>

#include <ColorMapping.h>

#include <VtkRenderEngine.h>
#include <VtkFluidVisualModule.h>

#include <ImColorbar.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(1.5, 1, 1.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));

	auto root = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	root->loadCube(Vec3f(-0.5, 0, -0.5), Vec3f(1.5, 2, 1.5), 0.02, true);
	root->loadSDF("../../data/bowl/bowl.sdf", false);

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->loadParticles(Vec3f(0.5, 0.2, 0.4), Vec3f(0.7, 1.5, 0.6), 0.005);
	root->addParticleSystem(fluid);

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);

	fluid->stateVelocity()->connect(calculateNorm->inVec());
	calculateNorm->outNorm()->connect(colorMapper->inScalar());

	fluid->graphicsPipeline()->pushModule(calculateNorm);
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto fRender = std::make_shared<VtkFluidVisualModule>();
	//fRender->setColor(1, 0, 0);
	fluid->stateTopology()->connect(fRender->inPointSet());
	fluid->graphicsPipeline()->pushModule(fRender);

	return scn;
}

int main()
{
	QtApp window;
	window.setRenderEngine(std::make_shared<VtkRenderEngine>());
	window.setSceneGraph(createScene());
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}