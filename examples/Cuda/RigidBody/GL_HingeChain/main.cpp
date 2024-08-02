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
	Real scale = 2.5;
	SphereInfo sphere;
	sphere.center = scale * Vec3f(-4.6, 20, 0.5);
	sphere.radius = scale * 2.5;
	//auto sphereActor = rigid->addSphere(sphere, rigidBody, 0.001);
	BoxInfo newbox, oldbox;
	oldbox.center = scale * Vec3f(-2.0, 20, 0.5);
	oldbox.halfLength = scale * Vec3f(0.05, 0.09, 0.02);
	oldbox.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
	rigidBody.linearVelocity = Vec3f(0, 0, 0);
	auto oldBoxActor = rigid->addBox(oldbox, rigidBody);
	rigidBody.linearVelocity = Vec3f(0, 0, 0);
	/*auto& hingeJoint1 = rigid->createHingeJoint(oldBoxActor, sphereActor);
	hingeJoint1.setAnchorPoint((oldbox.center + sphere.center) / 2);
	hingeJoint1.setAxis(Vec3f(0, 0, 1));*/

	for (int i = 0; i < 20; i++)
	{
		newbox.center = oldbox.center + scale * Vec3f(0.2, 0, 0);
		newbox.halfLength = oldbox.halfLength;
		newbox.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		auto newBoxActor = rigid->addBox(newbox, rigidBody);
		auto& hingeJoint = rigid->createHingeJoint(oldBoxActor, newBoxActor);
		hingeJoint.setAnchorPoint((oldbox.center + newbox.center) / 2);
		hingeJoint.setAxis(Vec3f(0, 0, 1));
		hingeJoint.setRange(-M_PI, M_PI);
		oldbox = newbox;
		oldBoxActor = newBoxActor;

		if (i == 19)
		{
			auto& pointJoint = rigid->createPointJoint(newBoxActor);
			pointJoint.setAnchorPoint(newbox.center);
		}
	}

	
	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(0.204, 0.424, 0.612));
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

	//Visualize Anchor point for joint
	/*auto anchorPointMapper = std::make_shared<AnchorPointToPointSet<DataType3f>>();
	rigid->stateCenter()->connect(anchorPointMapper->inCenter());
	rigid->stateRotationMatrix()->connect(anchorPointMapper->inRotationMatrix());
	rigid->stateTopology()->connect(anchorPointMapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(anchorPointMapper);

	auto pointRender2 = std::make_shared<GLPointVisualModule>();
	pointRender2->setColor(Color(1, 0, 0));
	pointRender2->varPointSize()->setValue(0.03f);
	anchorPointMapper->outPointSet()->connect(pointRender2->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender2);*/
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



