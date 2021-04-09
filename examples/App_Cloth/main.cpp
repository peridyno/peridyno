#include "GlutGUI/GLApp.h"

#include "Framework/SceneGraph.h"
#include "Framework/Log.h"

#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "PointRenderModule.h"
#include "SurfaceMeshRender.h"
#include "ParticleCloth.h"


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
	root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);
	root->loadShpere(Vector3f(0.5), 0.08f, 0.005f, false, true);

	std::shared_ptr<ParticleCloth<DataType3f>> child3 = std::make_shared<ParticleCloth<DataType3f>>();
	root->addParticleSystem(child3);

	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vector3f(1, 0.2, 1));
	child3->addVisualModule(m_pointsRender);
	child3->setVisible(false);

	child3->setMass(1.0);
  	child3->loadParticles("../../data/cloth/cloth.obj");
  	child3->loadSurface("../../data/cloth/cloth.obj");
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


