#include "GlutGUI/GLApp.h"

#include "Framework/SceneGraph.h"
#include "Topology/PointSet.h"
#include "Framework/Log.h"

#include "ParticleSystem/StaticBoundary.h"
#include "RigidBody/RigidBody.h"
#include "ParticleSystem/ParticleElastoplasticBody.h"
#include "ParticleSystem/GranularModule.h"

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
	root->loadSDF("../../data/bar/bar.sdf", false);
	root->translate(Vector3f(0.2f, 0.2f, 0));
	root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);

	std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child3 = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
	root->addParticleSystem(child3);

	auto m_pointsRender = std::make_shared<PointRenderModule>();

	m_pointsRender->setColor(Vector3f(0.98, 0.85, 0.40));
	child3->addVisualModule(m_pointsRender);

	child3->setMass(1.0);
  	child3->loadParticles("../../data/bunny/bunny_points.obj");
  	child3->loadSurface("../../data/bunny/bunny_mesh.obj");
	child3->translate(Vector3f(0.3f, 0.4f, 0.5f));
	child3->setDt(0.001f);
	auto elasto = std::make_shared<GranularModule<DataType3f>>();
	elasto->enableFullyReconstruction();
	child3->setElastoplasticitySolver(elasto);
	elasto->setCohesion(0);

	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	root->addRigidBody(rigidbody);
	rigidbody->loadShape("../../data/bar/bar.obj");
	rigidbody->setActive(false);
	rigidbody->translate(Vector3f(0.2f, 0.2f, 0));
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


