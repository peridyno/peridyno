#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GlutGUI/GLApp.h"

#include "Framework/SceneGraph.h"
#include "Topology/PointSet.h"
#include "Framework/Log.h"

#include "ParticleSystem/ParticleElastoplasticBody.h"
#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/ElasticityModule.h"
#include "RigidBody/RigidBody.h"

#include "SurfaceMeshRender.h"
#include "PointRenderModule.h"

using namespace std;
using namespace dyno;

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
// 	root->loadSDF("../data/bar/bar.sdf", false);
// 	root->translate(Vector3f(0.2f, 0.2f, 0));
	root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);

	std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child3 = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
	root->addParticleSystem(child3);

	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vector3f(0, 1, 1));
	child3->addVisualModule(ptRender);

	child3->setVisible(false);
	child3->setMass(1.0);
  	child3->loadParticles(Vector3f(-1.1), Vector3f(1.15), 0.1);
  	child3->loadSurface("../../data/standard/standard_cube20.obj");
	child3->scale(0.05);
	child3->translate(Vector3f(0.3, 0.2, 0.5));
	child3->getSurfaceNode()->setVisible(true);

	auto sRender = std::make_shared<SurfaceMeshRender>();
	child3->getSurfaceNode()->addVisualModule(sRender);
	sRender->setColor(Vector3f(1, 1, 1));


	std::shared_ptr<ParticleElasticBody<DataType3f>> child2 = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(child2);
	
	auto ptRender2 = std::make_shared<PointRenderModule>();
	ptRender2->setColor(Vector3f(0, 1, 1));
	child2->addVisualModule(ptRender2);

	child2->setVisible(false);
	child2->setMass(1.0);
	child2->loadParticles(Vector3f(-1.1), Vector3f(1.15), 0.1);
	child2->loadSurface("../../data/standard/standard_cube20.obj");
	child2->scale(0.05);
	child2->translate(Vector3f(0.5, 0.2, 0.5));
	child2->getElasticitySolver()->setIterationNumber(10);

	auto sRender2 = std::make_shared<SurfaceMeshRender>();
	child2->getSurfaceNode()->addVisualModule(sRender2);
	sRender2->setColor(Vector3f(1, 1, 0));
}


int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


