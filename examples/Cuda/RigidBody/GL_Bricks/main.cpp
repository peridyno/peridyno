#include <UbiApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include "Collision/NeighborElementQuery.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(1.0, 0, 0);

	BoxInfo box;
	box.halfLength = 0.5f * Vec3f(0.065, 0.065, 0.1);

	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			rigidBody.position = 0.5f * Vec3f(0.5f, 1.1 - 0.13 * i, 0.12f + 0.2 * j + 0.1 * (8 - i));
			
			auto boxAt = rigid->addBox(box, rigidBody);
		}

	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			rigidBody.position = 0.5f * Vec3f(2.5f, 1.1 - 0.13 * i, 0.12f + 0.2 * j + 0.1 * (8 - i));
			rigidBody.friction = 0.1;
			auto boxAt = rigid->addBox(box, rigidBody);
		}



	SphereInfo sphere;
	sphere.radius = 0.025f;

	RigidBodyInfo rigidSphere;
	rigidSphere.position = Vec3f(0.5f, 0.75f, 0.5f);
	auto sphereAt1 = rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.5f, 0.95f, 0.5f);
	auto sphereAt2 = rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.5f, 0.65f, 0.5f);
	sphere.radius = 0.05f;
	auto sphereAt3 = rigid->addSphere(sphere, rigidSphere);


	rigidSphere.position = Vec3f(0.0f);
	TetInfo tet;
	tet.v[0] = Vec3f(0.5f, 1.1f, 0.5f);
	tet.v[1] = Vec3f(0.5f, 1.2f, 0.5f);
	tet.v[2] = Vec3f(0.6f, 1.1f, 0.5f);
	tet.v[3] = Vec3f(0.5f, 1.1f, 0.6f);

	RigidBodyInfo rigidTet;
	rigidTet.position = (tet.v[0] + tet.v[1] + tet.v[2] + tet.v[3]) / 4;
	tet.v[0] -= rigidTet.position;
	tet.v[1] -= rigidTet.position;
	tet.v[2] -= rigidTet.position;
	tet.v[3] -= rigidTet.position;
	rigid->addTet(tet, rigidTet);
	auto TetAt = rigid->addTet(tet, rigidTet);




	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(0.5f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	//TODO: to enable using internal modules inside a node
	//Visualize contact normals
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->stateTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

	auto contactMapper = std::make_shared<ContactsToEdgeSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactMapper->inContacts());
	contactMapper->varScale()->setValue(0.02);
	rigid->graphicsPipeline()->pushModule(contactMapper);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Color(0, 0, 1));
	contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	//Visualize contact points
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Color(1, 0, 0));
	pointRender->varPointSize()->setValue(0.003f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	return scn;
}

int main()
{
	UbiApp app;
	app.setSceneGraph(creatBricks());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


