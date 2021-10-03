#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <Module/CalculateNorm.h>
#include <Quat.h>

#include <GLRenderEngine.h>
#include <GLElementVisualModule.h>
#include <ColorMapping.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

void creat_scene_fluid()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<RigidBodySystem<DataType3f>> rigid = scene.createNewScene<RigidBodySystem<DataType3f>>();

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.5, 0, 0);
	BoxInfo box;
	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			box.center = 0.5f * Vec3f(0.5f, 1.1 - 0.13 * i, 0.12f + 0.2 * j + 0.1 * (8 - i));
			box.halfLength = 0.5f * Vec3f(0.065, 0.065, 0.1);
			rigid->addBox(box, rigidBody);
		}

	SphereInfo sphere;
	sphere.center = Vec3f(0.5f, 0.75f, 0.5f);
	sphere.radius = 0.025f;

	RigidBodyInfo rigidSphere;
	rigid->addSphere(sphere, rigidSphere);

	sphere.center = Vec3f(0.5f, 0.95f, 0.5f);
	sphere.radius = 0.025f;
	rigid->addSphere(sphere, rigidSphere);

	sphere.center = Vec3f(0.5f, 0.65f, 0.5f);
	sphere.radius = 0.05f;
	rigid->addSphere(sphere, rigidSphere);

	TetInfo tet;
	tet.v[0] = Vec3f(0.5f, 1.1f, 0.5f);
	tet.v[1] = Vec3f(0.5f, 1.2f, 0.5f);
	tet.v[2] = Vec3f(0.6f, 1.1f, 0.5f);
	tet.v[3] = Vec3f(0.5f, 1.1f, 0.6f);
	rigid->addTet(tet, rigidSphere);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->currentTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);
// 	auto eRender = std::make_shared<GLElementVisualModule>();
// 	eRender->discreteSet = DE;
// 	rigid->varTimeStep()->connect(eRender->inTimeStep());
// 	
// 	//rigid->addVisualModule(eRender);
// 	eRender->setColor(Vec3f(1, 0, 0));
// 
// 	rigid->graphicsPipeline()->pushModule(eRender);
	
	//rigid->initialize();
	

	GLRenderEngine* engine = new GLRenderEngine;

	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1280, 768);
	window.mainLoop();

	delete engine;
}



int main()
{

	creat_scene_fluid();
	//test_render();

	return 0;
}


