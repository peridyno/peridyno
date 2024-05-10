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
#include <Mapping/AnchorPointToPointSet.h>

#include "Collision/NeighborElementQuery.h"


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());
	
	RigidBodyInfo rigidBody;
	BoxInfo newbox, oldbox;
	oldbox.center = Vec3f(-2.0, 4, 0.5);
	oldbox.halfLength = Vec3f(0.02, 0.09, 0.02);
	oldbox.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
	rigidBody.linearVelocity = Vec3f(0, 0, 0);
	auto oldBoxActor = rigid->addBox(oldbox, rigidBody);
	rigidBody.linearVelocity = Vec3f(0, 0, 0);
	for (int i = 0; i < 30; i++)
	{
		newbox.center = oldbox.center + Vec3f(0.2, 0, 0);
		newbox.halfLength = oldbox.halfLength;
		newbox.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		auto newBoxActor = rigid->addBox(newbox, rigidBody);
		auto& hingeJoint = rigid->createHingeJoint(oldBoxActor, newBoxActor);
		hingeJoint.setAnchorPoint((oldbox.center + newbox.center) / 2, oldbox.center, newbox.center, oldbox.rot, newbox.rot);
		hingeJoint.setAxis(Vec3f(0, 0, 1), oldbox.rot, newbox.rot);
		hingeJoint.setRange(-M_PI / 2, M_PI / 2);
		oldbox = newbox;
		oldBoxActor = newBoxActor;

		if (i == 29)
		{
			auto& pointJoint = rigid->createPointJoint(newBoxActor);
			pointJoint.setAnchorPoint(newbox.center);
		}
	}

	BoxInfo box;
	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			box.center = 0.5f * Vec3f(0.5f, 1.1 - 0.13 * i, 0.12f + 0.21 * j + 0.1 * (8 - i));
			box.halfLength = 0.5f * Vec3f(0.065, 0.065, 0.1);
			rigid->addBox(box, rigidBody);
		}
	
	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(1.0f);
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
	GlfwApp app;
	app.setSceneGraph(creatBricks());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


