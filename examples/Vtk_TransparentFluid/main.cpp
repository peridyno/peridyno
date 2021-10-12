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

void CreateScene(AppBase* app)
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vec3f(1.5, 1, 1.5));
	scene.setLowerBound(Vec3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(-0.5, 0, -0.5), Vec3f(1.5, 2, 1.5), 0.02, true);
	root->loadSDF("../../data/bowl/bowl.sdf", false);

	std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
	fluid->loadParticles(Vec3f(0.5, 0.2, 0.4), Vec3f(0.7, 1.5, 0.6), 0.005);
	fluid->setMass(100);
	root->addParticleSystem(fluid);

	// point
	auto fRender = std::make_shared<VtkFluidVisualModule>();
	//fRender->setColor(1, 0, 0);
	fluid->currentTopology()->connect(fRender->inPointSet());
	fluid->graphicsPipeline()->pushModule(fRender);
}

int main()
{
	RenderEngine* engine = new VtkRenderEngine;

	GlfwApp window;
	window.setRenderEngine(engine);

	CreateScene(&window);

	// window.createWindow(2048, 1152);
	window.createWindow(1024, 768);
	window.mainLoop();
	
	delete engine;

	return 0;
}


