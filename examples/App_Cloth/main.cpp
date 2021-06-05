#include "GlfwGUI/GlfwApp.h"

#include "Framework/SceneGraph.h"
#include "Framework/Log.h"

#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
//#include "PointRenderModule.h"
//#include "SurfaceMeshRender.h"
#include "ParticleCloth.h"

#include "RenderEngine.h"
#include "module/VtkSurfaceVisualModule.h"
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
	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);
	root->loadShpere(Vec3f(0.5), 0.08f, 0.005f, false, true);

	std::shared_ptr<ParticleCloth<DataType3f>> child3 = std::make_shared<ParticleCloth<DataType3f>>();
	root->addParticleSystem(child3);

	child3->setMass(1.0);
	child3->loadParticles("../../data/cloth/cloth.obj");
	child3->loadSurface("../../data/cloth/cloth.obj");

	auto sRender = std::make_shared<SurfaceVisualModule>();
	sRender->setColor(0.4, 0.75, 1);
	child3->getSurfaceNode()->addVisualModule(sRender);

	auto pRender = std::make_shared<PointVisualModule>();
	pRender->setColor(1, 0.2, 1);
	child3->addVisualModule(pRender);
	child3->setVisible(true);

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
	window.mUseNewRenderEngine = true;

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


