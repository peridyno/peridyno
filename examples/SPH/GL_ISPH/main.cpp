#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>

#include <ParticleSystem/ParticleFluid.h>
#include "ParticleSystem/GhostParticles.h"
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/IncompressibleFluid.h>

#include <RigidBody/RigidBody.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <ColorMapping.h>
#include <ImColorbar.h>

using namespace std;
using namespace dyno;

bool useVTK = false;

void CreateScene(AppBase* app)
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vec3f(0.5, 1, 0.5));
	scene.setLowerBound(Vec3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(-0.1f, 0.0f, -0.1f), Vec3f(0.1f, 1.0f, 0.1f), 0.02, true);
	//root->loadSDF(getAssetPath() + "bowl/bowl.sdf", false);

	std::shared_ptr<ParticleSystem<DataType3f>> fluid = std::make_shared<ParticleSystem<DataType3f>>();
	fluid->loadParticles(Vec3f(-0.0, 0.0, -0.1), Vec3f(0.1, 0.1, 0.1), 0.005);

	auto ghost = std::make_shared<GhostParticles<DataType3f>>();
	ghost->loadPlane();

	auto incompressibleFluid = std::make_shared<IncompressibleFluid<DataType3f>>();
	incompressibleFluid->setFluidParticles(fluid);
	incompressibleFluid->setBoundaryParticles(ghost);

	root->addAncestor(incompressibleFluid);
	root->addParticleSystem(fluid);

	{
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(5.0f);

		fluid->currentVelocity()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());

		fluid->graphicsPipeline()->pushModule(calculateNorm);
		fluid->graphicsPipeline()->pushModule(colorMapper);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Vec3f(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		ptRender->setColorMapRange(0, 5);

		fluid->currentTopology()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		fluid->graphicsPipeline()->pushModule(ptRender);

		// A simple color bar widget for node
		auto colorBar = std::make_shared<ImColorbar>();
		colorBar->varMax()->setValue(5.0f);
		calculateNorm->outNorm()->connect(colorBar->inScalar());
		// add the widget to app
		fluid->graphicsPipeline()->pushModule(colorBar);
	}
	
	{
		auto ghostRender = std::make_shared<GLPointVisualModule>();
		ghostRender->setColor(Vec3f(1, 0, 0));
		ghostRender->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);

		ghost->currentTopology()->connect(ghostRender->inPointSet());

		ghost->graphicsPipeline()->pushModule(ghostRender);
	}
}

int main()
{
	RenderEngine* engine = new GLRenderEngine;

	GlfwApp window;
	window.setRenderEngine(engine);

	CreateScene(&window);

	// window.createWindow(2048, 1152);
	window.createWindow(1024, 768);
	window.mainLoop();
	
	delete engine;

	return 0;
}


