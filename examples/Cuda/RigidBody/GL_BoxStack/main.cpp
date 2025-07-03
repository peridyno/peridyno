#include <GlfwApp.h>

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

std::shared_ptr<SceneGraph> createBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-200.0f, 0.0f, -200.0f));
	scn->setUpperBound(Vec3f(200.0f, 400.0f, 200.0f));
	//	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	Real dHat = 0.001f;
	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());
	rigid->setDt(0.016f);
	bool hasBall = true;
	uint x = 10;
	uint y = 10;
	uint z = 10;
	float hl_x = 1.0f;
	float hl_z = 1.0f;
	float hl_y = 1.0f;
	float interval_x = 0.001f;
	float interval_y = 0.001f;
	float offset_x = 0.0f;
	float offset_y = 0.0f;

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	BoxInfo box;
	box.halfLength = Vec3f(hl_x, hl_z, hl_y);
	int id = 0;
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < z; j++)
		{
			for (int k = 0; k < y; k++)
			{
				rigidBody.position = Vec3f(2 * i * (hl_x + dHat + interval_x) - (hl_x + dHat + interval_x) * x + (j % 2 ? offset_x : 0.0f), \
					2 * j * hl_z + hl_z, \
					2 * k * (hl_y + dHat + interval_y) - (hl_y + dHat + interval_y) * y + (j % 2 ? offset_y : 0.0f));
				rigidBody.bodyId = id++;
				auto boxAt = rigid->addBox(box, rigidBody, 100.0f);
			}
		}
	}

	if (hasBall)
	{
		rigidBody.position = Vec3f(0.0f, 15.0f, 40.0f);
		rigidBody.linearVelocity = Vec3f(0.0f, 0.0f, -100.0f);
		SphereInfo sphere;
		sphere.radius = 3.0f;
		auto sphereAt = rigid->addSphere(sphere, rigidBody, 100.0f);
	}

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color::SteelBlue2());
	sRender->setAlpha(1.0f);
	//sRender->setMetallic(1.0f);
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
	wireRender->setColor(Color(0, 0, 0));
	mapper->outTriangleSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	//Visualize contact points
	/*auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Color(1, 0, 0));
	pointRender->varPointSize()->setValue(0.003f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);*/

	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(createBricks());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


