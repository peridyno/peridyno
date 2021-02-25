#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GlutGUI/GLApp.h"

#include "Framework/SceneGraph.h"
#include "Framework/Log.h"

#include "ParticleSystem/ParticleFluid.h"
#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"

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
	scene.setUpperBound(Vector3f(1.5, 1, 1.5));
	scene.setLowerBound(Vector3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(-0.5, 0, -0.5), Vector3f(1.5, 2, 1.5), 0.02, true);
	root->loadSDF("../../data/bowl/bowl.sdf", false);

	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(child1);

	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vector3f(1, 0, 0));
	ptRender->setColorRange(0, 4);
	child1->addVisualModule(ptRender);

	//child1->loadParticles("../data/fluid/fluid_point.obj");
	child1->loadParticles(Vector3f(0.5, 0.2, 0.4), Vector3f(0.7, 1.5, 0.6), 0.005);
	child1->setMass(100);
	child1->currentVelocity()->connect(&ptRender->m_vecIndex);

	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	root->addRigidBody(rigidbody);
	rigidbody->loadShape("../../data/bowl/bowl.obj");
	rigidbody->setActive(false);
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


