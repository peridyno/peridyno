#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/IncompressibleModel.h>

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
	scene.setUpperBound(Vec3f(1.5, 1, 1.5));
	scene.setLowerBound(Vec3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(0.0f), Vec3f(1.0f), 0.02, true);
	//root->loadSDF(getAssetPath() + "bowl/bowl.sdf", false);

	std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
	fluid->loadParticles(Vec3f(0.5, 0.05, 0.5), Vec3f(0.55, 0.1, 0.55), 0.005);
	fluid->animationPipeline()->clear();

	auto model = std::make_shared<IncompressibleModel<DataType3f>>();
	model->m_smoothingLength.setValue(0.01);
	fluid->varTimeStep()->connect(model->inTimeStep());
	fluid->currentPosition()->connect(model->inPosition());
	fluid->currentVelocity()->connect(model->inVelocity());
	fluid->currentForce()->connect(model->inForce());
	fluid->animationPipeline()->pushModule(model);

	root->addParticleSystem(fluid);

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

#include "ParticleSystem/ParticleApproximation.h"

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


