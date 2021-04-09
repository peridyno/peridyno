#include "GlutGUI/GLApp.h"
#include "Framework/SceneGraph.h"
#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/ElasticityModule.h"

#include "SurfaceMeshRender.h"
#include "PointRenderModule.h"

using namespace dyno;

int main()
{
	Log::sendMessage(Log::Info, "Simulation start");

	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);

	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vector3f(0, 1, 1));
	bunny->addVisualModule(m_pointsRender);

	bunny->setMass(1.0);
	bunny->loadParticles("../../data/bunny/bunny_points.obj");
	bunny->loadSurface("../../data/bunny/bunny_mesh.obj");
	bunny->scale(1.0f);
	bunny->translate(Vector3f(0.5f, 0.1f, 0.5f));
	bunny->setVisible(true);

	auto sRender = std::make_shared<SurfaceMeshRender>();
	bunny->getSurfaceNode()->addVisualModule(sRender);
	sRender->setColor(Vector3f(1, 1, 0));

	bunny->getElasticitySolver()->setIterationNumber(10);
	bunny->getElasticitySolver()->inHorizon()->setValue(0.01);

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	return 0;
}


