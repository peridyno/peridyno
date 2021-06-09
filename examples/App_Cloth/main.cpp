#include "GlfwGUI/GlfwApp.h"

#include "Framework/SceneGraph.h"
#include "Framework/Log.h"

#include "Peridynamics/ParticleElasticBody.h"

#include "ParticleSystem/StaticBoundary.h"

#include "PointRenderModule.h"
#include "SurfaceMeshRender.h"
#include "ParticleCloth.h"

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);
	root->loadShpere(Vec3f(0.5, 0.7f, 0.5), 0.08f, 0.005f, false, true);

	std::shared_ptr<ParticleCloth<DataType3f>> child3 = std::make_shared<ParticleCloth<DataType3f>>();
	root->addParticleSystem(child3);

	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vec3f(1, 0.2, 1));
	child3->addVisualModule(m_pointsRender);
	child3->setVisible(false);

	child3->setMass(1.0);
  	child3->loadParticles("../../data/cloth/cloth.obj");
  	child3->loadSurface("../../data/cloth/cloth.obj");
}

int main()
{
	CreateScene();

	GlfwApp window;
	window.createWindow(1024, 768);

	window.mainLoop();
	return 0;
}


