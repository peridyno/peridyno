#include "GlfwGUI/GlfwApp.h"

#include "Framework/SceneGraph.h"
#include "Framework/Log.h"

#include "Peridynamics/ParticleElasticBody.h"

#include "ParticleSystem/StaticBoundary.h"

#include "RigidBody/RigidBody.h"

#include "PointRenderModule.h"
#include "SurfaceMeshRender.h"
#include "ParticleCloth.h"

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadSDF("../../data/t-shirt/body-scaled.sdf");
// 	root->loadCube(Vec3f(-1.5, 0, -1.5), Vec3f(1.5, 3.5, 1.5), 0.05f, true);
// 	root->loadShpere(Vec3f(0.0, 0.7f, 0.0), 0.15f, 0.005f, false, true);

	std::shared_ptr<ParticleCloth<DataType3f>> child3 = std::make_shared<ParticleCloth<DataType3f>>();
	root->addParticleSystem(child3);

	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vec3f(1, 0.2, 1));
	child3->addVisualModule(m_pointsRender);
	child3->setVisible(false);

	child3->setMass(1.0);
//   	child3->loadParticles("../../data/cloth/cloth.obj");
//   	child3->loadSurface("../../data/cloth/cloth.obj");

	child3->loadParticles("../../data/t-shirt/t-shirt.obj");
	child3->loadSurface("../../data/t-shirt/t-shirt.obj");

	//child3->scale(0.5);

	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	root->addRigidBody(rigidbody);
	rigidbody->loadShape("../../data/t-shirt/body-scaled.obj");
	rigidbody->scale(0.95);
	rigidbody->translate(Vec3f(0, 0.07, 0));
	rigidbody->setActive(false);

	auto render = std::make_shared<SurfaceMeshRender>();
	render->setColor(Vec3f(0.1, 1.0, 1));
	rigidbody->getSurface()->addVisualModule(render);
}

int main()
{
	CreateScene();

	GlfwApp window;
	window.createWindow(1024, 768);

	window.mainLoop();
	return 0;
}


