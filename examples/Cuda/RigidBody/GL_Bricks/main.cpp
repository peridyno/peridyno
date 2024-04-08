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

std::shared_ptr<SceneGraph> creatBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	BoxInfo box1, box2;
	box1.center = Vec3f(0, 0.2, 0);
	box1.halfLength = Vec3f(0.1);

	box2.center = Vec3f(0, 0.6, 0);
	box2.halfLength = Vec3f(0.1);

	HingeJoint<Real> joint(0, 1);
	joint.setAnchorPoint(box1.center + Vec3f(0.0, 0.2f, 0.0), box1.center, box2.center, box1.rot, box2.rot);
	joint.setAxis(Vec3f(1, 0, 0), box1.rot, box2.rot);
	rigid->addHingeJoint(joint);
	rigid->addBox(box1, rigidBody);
	rigidBody.angularVelocity = Vec3f(0.0, 0, 10.0);
	rigid->addBox(box2, rigidBody);

	/*for (int i = 0; i < 20; i++)
	{
		box.center = Vec3f(0.5f, 0.1f + i * 0.2f, 0.5f);
		box.halfLength = Vec3f(0.1f);
		if (i != 20)
		{
			HingeJoint<Real> joint(i, i + 1);
			joint.setAnchorPoint(box.center + Vec3f(0.0, 0.1f, 0.0), box.center, box.center + Vec3f(0, 0.2, 0), box.rot, box.rot);
			joint.setAxis(Vec3f(0, 0, 1), box.rot, box.rot);
			rigid->addHingeJoint(joint);
		}
		rigid->addBox(box, rigidBody);
	}*/




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
	GlfwApp app;
	app.setSceneGraph(creatBricks());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


