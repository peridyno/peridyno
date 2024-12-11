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
#include "Collision/CollistionDetectionBoundingBox.h"

using namespace std;
using namespace dyno;

/**
 * This example demonstrate how to simulate rigid bodies as a compound
 */

std::shared_ptr<SceneGraph> creatRigidCompound()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());
	
	auto actor = rigid->createRigidBody(Vec3f(0.0f, 0.3f, 0.0f), Quat<float>(0.5, Vec3f(1.0f, 0.0f, 1.0f)));

	BoxInfo box;
	box.center = Vec3f(0.15f, 0.0f, 0.0f);
	box.halfLength = Vec3f(0.05f);
	rigid->bindBox(actor, box);

	SphereInfo sphere;
	sphere.center = Vec3f(0.0f, 0.0f, 0.1f);
	sphere.radius = 0.1f;
	rigid->bindSphere(actor, sphere);

	CapsuleInfo capsule;
	capsule.center = Vec3f(-0.15f, 0.0f, 0.0f);
	capsule.radius = 0.1f;
	capsule.halfLength = 0.1f;
	rigid->bindCapsule(actor, capsule);

	auto actor2 = rigid->createRigidBody(Vec3f(-0.1f, 0.6f, 0.0f), Quat<float>());
	SphereInfo sphere2;
	sphere2.center = Vec3f(0.0f, 0.0f, 0.0f);
	sphere2.radius = 0.1f;
	rigid->bindSphere(actor2, sphere2);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(0.5f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	//Visualize contact points
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->stateTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Color(1, 0, 0));
	pointRender->varPointSize()->setValue(0.01f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

// 	//Visualize contact points
// 	auto cdBV = std::make_shared<CollistionDetectionBoundingBox<DataType3f>>();
// 	rigid->stateTopology()->connect(cdBV->inDiscreteElements());
// 	//jeep->inTriangleSet()->connect(cdBV->inTriangleSet());
// 	rigid->graphicsPipeline()->pushModule(cdBV);
// 
// 	auto boundaryContactsMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
// 	cdBV->outContacts()->connect(boundaryContactsMapper->inContacts());
// 	rigid->graphicsPipeline()->pushModule(boundaryContactsMapper);
// 
// 	auto boundaryContactsRender = std::make_shared<GLPointVisualModule>();
// 	boundaryContactsRender->setColor(Color(0, 1, 0));
// 	boundaryContactsRender->varPointSize()->setValue(0.01f);
// 	boundaryContactsMapper->outPointSet()->connect(boundaryContactsRender->inPointSet());
// 	rigid->graphicsPipeline()->pushModule(boundaryContactsRender);

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_GLFW);
	app.setSceneGraph(creatRigidCompound());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


