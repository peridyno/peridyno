#include "GlfwGUI/GlfwApp.h"

#include "Framework/SceneGraph.h"
#include "Framework/Log.h"

#include "ParticleSystem/ParticleFluid.h"
#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"

#include "PointRenderModule.h"

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vec3f(1.5, 1, 1.5));
	scene.setLowerBound(Vec3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(-0.5, 0, -0.5), Vec3f(1.5, 2, 1.5), 0.02, true);
	root->loadSDF("../../data/bowl/bowl.sdf", false);

	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(child1);

	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vec3f(1, 0, 0));
	ptRender->setColorRange(0, 4);
	child1->addVisualModule(ptRender);

	child1->loadParticles(Vec3f(0.5, 0.2, 0.4), Vec3f(0.7, 1.5, 0.6), 0.005);
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

	GlfwApp window;
	window.createWindow(1024, 768);
	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


