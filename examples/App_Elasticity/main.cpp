#include "Framework/SceneGraph.h"
#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/ElasticityModule.h"

#include "../VTK/VtkApp/VtkApp.h"
#include "../VTK/VtkVisualModule/VtkSurfaceVisualModule.h"
#include "../VTK/VtkVisualModule/VtkPointVisualModule.h"

using namespace dyno;

int main()
{
	Log::sendMessage(Log::Info, "Simulation start");

	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);

	auto m_pointsRender = std::make_shared<PointVisualModule>();
	m_pointsRender->setColor(0, 1, 1);
	bunny->addVisualModule(m_pointsRender);

	bunny->setMass(1.0);
	bunny->loadParticles("../../data/bunny/bunny_points.obj");
	bunny->loadSurface("../../data/bunny/bunny_mesh.obj");
	bunny->scale(1.0f);
	bunny->translate(Vec3f(0.5f, 0.1f, 0.5f));
	bunny->setVisible(true);

	auto sRender = std::make_shared<SurfaceVisualModule>();
	bunny->getSurfaceNode()->addVisualModule(sRender);
	sRender->setColor(1, 1, 0);

	bunny->getElasticitySolver()->setIterationNumber(10);
	bunny->getElasticitySolver()->inHorizon()->setValue(0.01);

	scene.initialize();
	
	VtkApp window;
	window.createWindow(1024, 768);
	window.mainLoop();
	
	return 0;

}


