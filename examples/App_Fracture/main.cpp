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
#include "ParticleSystem/StaticBoundary.h"
#include "RigidBody/RigidBody.h"
#include "ParticleSystem/FractureModule.h"

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
	scene.setLowerBound(Vector3f(0, 0, 0));
	scene.setUpperBound(Vector3f(1, 0.5, 0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
// 	root->loadSDF("../data/bar/bar.sdf", false);
// 	root->translate(Vector3f(0.2f, 0.2f, 0));
	root->loadCube(Vector3f(0), Vector3f(1, 0.5, 0.5), 0.005, true);
	root->loadCube(Vector3f(-0.1, 0, -0.1), Vector3f(0.1, 0.25, 0.6), 0.005, false);

	std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child3 = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
	root->addParticleSystem(child3);
	
	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vector3f(0, 1, 1));
	child3->addVisualModule(m_pointsRender);

	child3->setMass(1.0);
  	//child3->loadParticles("../data/bunny/bunny_points.obj");
  	//child3->loadSurface("../data/bunny/bunny_mesh.obj");
	child3->loadParticles(Vector3f(0, 0.25, 0.1), Vector3f(0.2f, 0.4, 0.4), 0.005f);
	//child3->translate(Vector3f(0.3, 0.4, 0.5));
	auto fracture = std::make_shared<FractureModule<DataType3f>>();
	child3->setElastoplasticitySolver(fracture);

// 	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
// 	root->addRigidBody(rigidbody);
// 	rigidbody->loadShape("../data/bar/bar.obj");
// 	rigidbody->setActive(false);
// 	rigidbody->translate(Vector3f(0.2f, 0.2f, 0));
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


