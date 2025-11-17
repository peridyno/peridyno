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

	//box.halfLength = Vec3f(0.1, 0.1, 0.1);

	/*for (int i = 0; i < 5; i++)
	{
		rigidBody.position = Vec3f(0.5f, 0.11, i * 0.3);
		rigidBody.linearVelocity = Vec3f(1.0, 0, 0);
		rigidBody.friction = 0.02 * (i+1);
		auto boxAt = rigid->addBox(box, rigidBody, 100);
	}*/


	Real boardLength = 4.0f;
	Real boardWidth = 0.5f;
	Real boardThickness = 0.02f;
	Real boardAngle = M_PI / 5.0f;
	Real boxEdgeLength = 0.2f;
	Real d = boardLength * 0.75f;

	RigidBodyInfo rigidBody;
	rigidBody.friction = 0.9f;
	BoxInfo box;

	box.halfLength = Vec3f(0.5f * boardLength, boardThickness, boardWidth);
	box.rot = Quat1f(boardAngle, Vec3f(0, 0, 1).normalize());
	rigidBody.position = Vec3f(0.0f, 0.5f * boardLength * sinf(boardAngle), 0.0f);
	rigidBody.motionType = BodyType::Dynamic;
	auto & ac = rigid->addBox(box, rigidBody, 10);
	auto& fixedJoint = rigid->createUnilateralFixedJoint(ac);
	fixedJoint.setAnchorPoint(rigidBody.position);

	box.halfLength = Vec3f(boxEdgeLength / 2.0f);

	rigidBody.position = Vec3f(cosf(boardAngle) * ((0.5f * boxEdgeLength + boardThickness + 0.0001f) / tanf(boardAngle) + d) - (0.5f * boxEdgeLength + boardThickness + 0.0001f) / sinf(boardAngle) - 0.5f * cosf(boardAngle) * boardLength, \
		sinf(boardAngle) * ((0.5f * boxEdgeLength + boardThickness + 0.0001f) / tanf(boardAngle) + d), \
		0.0f);

	box.rot = Quat1f(boardAngle, Vec3f(0, 0, 1).normalize());
	rigidBody.bodyId = 1;
	rigidBody.motionType = BodyType::Dynamic;
	rigid->addBox(box, rigidBody, 1000);

	


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
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Color(1, 0, 0));
	pointRender->varPointSize()->setValue(0.03f);
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


