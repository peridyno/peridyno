#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>

#include <ParticleSystem/ParticleFluid.h>
#include <RigidBody/RigidBody.h>
#include <ParticleSystem/StaticBoundary.h>

#include <VtkRenderEngine.h>
#include <VtkFluidVisualModule.h>

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

	// point
	auto fRender = std::make_shared<VtkFluidVisualModule>();
	//fRender->setColor(1, 0, 0);
	fluid->stateTopology()->connect(fRender->inPointSet());
	fluid->graphicsPipeline()->pushModule(fRender);

	return scn;
}

int main()
{
	GlfwApp window;
	window.setRenderEngine(std::make_shared<VtkRenderEngine>());
	window.setSceneGraph(createScene());
	// window.createWindow(2048, 1152);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}


