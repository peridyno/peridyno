#include "GlfwGUI/GlfwApp.h"

#include "Framework/SceneGraph.h"
#include "Topology/PointSet.h"
#include "Framework/Log.h"

#include "ParticleSystem/StaticBoundary.h"
#include "RigidBody/RigidBody.h"

//#include "PointRenderModule.h"

#include "ParticleViscoplasticBody.h"

#include "RenderEngine.h"
#include "module/VtkPointVisualModule.h"

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
	root->loadCube(Vec3f(0), Vec3f(1), 0.005, true);

	for (size_t i = 0; i < 5; i++)
	{
		root->loadCube(Vec3f(0.2f + i * 0.08f, 0.2f, 0.0f), Vec3f(0.25 + i * 0.08f, 0.25f, 1.0f), 0.005f, false);
	}

	std::shared_ptr<ParticleViscoplasticBody<DataType3f>> child3 = std::make_shared<ParticleViscoplasticBody<DataType3f>>();
	root->addParticleSystem(child3);

	auto ptRender = std::make_shared<PointVisualModule>();
	ptRender->setColor(0, 1, 1);
	child3->addVisualModule(ptRender);

	child3->setMass(1.0);
  	child3->loadParticles("../../data/bunny/bunny_points.obj");
  	child3->loadSurface("../../data/bunny/bunny_mesh.obj");
	child3->translate(Vec3f(0.4, 0.4, 0.5));

	std::shared_ptr<ParticleViscoplasticBody<DataType3f>> child4 = std::make_shared<ParticleViscoplasticBody<DataType3f>>();
	root->addParticleSystem(child4);
	auto ptRender2 = std::make_shared<PointVisualModule>();
	ptRender2->setColor(1, 0, 1);
	child4->addVisualModule(ptRender2);

	child4->setMass(1.0);
	child4->loadParticles("../../data/bunny/bunny_points.obj");
	child4->loadSurface("../../data/bunny/bunny_mesh.obj");
	child4->translate(Vec3f(0.4, 0.4, 0.9));
}


int main()
{
	CreateScene();
	// we should use vtk GUI since some rendering is supported by OpenGL 3.2+
	SceneGraph::getInstance().initialize();
	RenderEngine engine;
	engine.setSceneGraph(&SceneGraph::getInstance());
	engine.start();
	return 0;

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GlfwApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


