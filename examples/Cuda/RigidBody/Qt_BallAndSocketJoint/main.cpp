#include <QtApp.h>

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

	BoxInfo newBox, oldBox;
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(1, 0.0, 1.0);
	oldBox.center = Vec3f(0, 0.1, 0);
	oldBox.halfLength = Vec3f(0.05, 0.05, 0.05);
	rigid->addBox(oldBox, rigidBody);

	

	for (int i = 0; i < 100; i++)
	{
		rigidBody.linearVelocity = Vec3f(0, 0, 0);
		newBox.center = oldBox.center + Vec3f(0.0, 0.12f, 0.0);
		newBox.halfLength = oldBox.halfLength;
		rigid->addBox(newBox, rigidBody);
		BallAndSocketJoint<Real> joint(i, i + 1);
		joint.setAnchorPoint(oldBox.center + Vec3f(0.0, 0.06f, 0.0), oldBox.center, newBox.center);
		rigid->addBallAndSocketJoint(joint);
		oldBox = newBox;
	}




	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(0.3f, 0.5f, 0.9f));
	sRender->setAlpha(0.8f);
	sRender->setRoughness(0.7f);
	sRender->setMetallic(3.0f);
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
	pointRender->varPointSize()->setValue(0.0003f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	//Visualize Anchor point for joint
	auto anchorPointMapper = std::make_shared<AnchorPointToPointSet<DataType3f>>();
	rigid->stateCenter()->connect(anchorPointMapper->inCenter());
	rigid->stateRotationMatrix()->connect(anchorPointMapper->inRotationMatrix());
	rigid->stateBallAndSocketJoints()->connect(anchorPointMapper->inBallAndSocketJoints());
	//rigid->stateSliderJoints()->connect(anchorPointMapper->inSliderJoints());
	//rigid->stateHingeJoints()->connect(anchorPointMapper->inHingeJoints());
	//rigid->stateFixedJoints()->connect(anchorPointMapper->inFixedJoints());
	rigid->graphicsPipeline()->pushModule(anchorPointMapper);

	auto pointRender2 = std::make_shared<GLPointVisualModule>();
	pointRender2->setColor(Color(1, 0, 0));
	pointRender2->varPointSize()->setValue(0.002f);
	anchorPointMapper->outPointSet()->connect(pointRender2->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender2);

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(creatBricks());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


