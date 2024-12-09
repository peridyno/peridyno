#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>

using namespace std;
using namespace dyno;

/**
 * @brief This is an example to demonstrate how the collision mask works
 * 
 */

std::shared_ptr<SceneGraph> creatScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.5, 0, 0);

	// Boxes are set to being able to collided with other boxes only
	rigidBody.collisionMask = CT_BoxOnly;
	BoxInfo box;
	box.halfLength = 0.5f * Vec3f(0.065, 0.065, 0.1);
	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			rigidBody.position = 0.5f * Vec3f(0.5f, 1.1 - 0.13 * i, 0.12f + 0.21 * j + 0.1 * (8 - i));
			rigid->addBox(box, rigidBody);
		}

	SphereInfo sphere;
	sphere.center = Vec3f(0.0f);
	sphere.radius = 0.025f;

	RigidBodyInfo rigidSphere;
	// Spheres are set to being able to collided with other spheres only
	rigidSphere.position = Vec3f(0.5f, 0.75f, 0.5f);
	rigidSphere.collisionMask = CT_SphereOnly;
	rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.5f, 0.95f, 0.5f);
	rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.5f, 0.65f, 0.5f);
	sphere.radius = 0.05f;
	rigid->addSphere(sphere, rigidSphere);

	TetInfo tet;
	RigidBodyInfo rigidTet;
	tet.v[0] = Vec3f(0.5f, 1.1f, 0.5f);
	tet.v[1] = Vec3f(0.5f, 1.2f, 0.5f);
	tet.v[2] = Vec3f(0.6f, 1.1f, 0.5f);
	tet.v[3] = Vec3f(0.5f, 1.1f, 0.6f);
	rigid->addTet(tet, rigidTet);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(creatScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


