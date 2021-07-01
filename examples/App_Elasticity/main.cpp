#include "GlfwGUI/GlfwApp.h"
#include "Framework/SceneGraph.h"

#include "ParticleSystem/StaticBoundary.h"

#include "Peridynamics/ParticleElasticBody.h"
#include "Peridynamics/ElasticityModule.h"

#include "module/SurfaceRender.h"
#include "module/PointRender.h"

using namespace dyno;

int main()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);

	bunny->setMass(1.0);
	bunny->loadParticles("../../data/bunny/bunny_points.obj");
	bunny->loadSurface("../../data/bunny/bunny_mesh.obj");
	bunny->scale(1.0f);
	bunny->translate(Vec3f(0.5f, 0.1f, 0.5f));
	bunny->setVisible(true);

	auto sRender = std::make_shared<SurfaceRenderer>();
	bunny->getSurfaceNode()->graphicsPipeline()->pushPersistentModule(sRender);
	sRender->setColor(Vec3f(1, 1, 0));

	bunny->getElasticitySolver()->setIterationNumber(10);
	bunny->getElasticitySolver()->inHorizon()->setValue(0.01);

	GlfwApp window;
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}


