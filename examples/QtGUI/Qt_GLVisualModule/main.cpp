#include <QtApp.h>
#include <GlfwApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/SquareEmitter.h>
#include <ParticleSystem/MakeParticleSystem.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

#include <ImColorbar.h>

#include "NodePortConnectionTest.h"
#include "InputFieldTest.h"


#include <RigidBody/RigidBodySystem.h>

#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include "Node/GLPointVisualNode.h"

#include "Collision/NeighborElementQuery.h"

#include "SphereModel.h"
#include "SphereSampler.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(1.5, 1, 1.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));

	//Create a sphere
	auto sphere = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	sphere->varLocation()->setValue(Vec3f(0.6, 0.85, 0.5));
	sphere->varRadius()->setValue(0.1f);
	sphere->graphicsPipeline()->disable();

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<SphereSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->graphicsPipeline()->disable();

	sphere->outSphere()->connect(sampler->inSphere());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());

	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	//fluid->loadParticles(Vec3f(0.5, 0.2, 0.4), Vec3f(0.7, 1.5, 0.6), 0.005);
	initialParticles->connect(fluid->importInitialStates());

	auto testNode = scn->addNode(std::make_shared<NodePortConnectionTest<DataType3f>>());
	fluid->connect(testNode->importParticleSystem());
 	
	auto ptVisulizer = scn->addNode(std::make_shared<GLPointVisualNode<DataType3f>>());

	auto outTop = fluid->statePointSet()->promoteOuput();
	auto outVel = fluid->stateVelocity()->promoteOuput();
	outTop->connect(ptVisulizer->inPoints());
	outVel->connect(ptVisulizer->inVector());

	auto nullNode = scn->addNode(std::make_shared<InputFieldTest<DataType3f>>());
	testNode->outPointSetOut()->connect(nullNode->inPointSet());

	return scn;
}

std::shared_ptr<SceneGraph> creatBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.5, 0, 0);
	BoxInfo box;
	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			box.center = 0.5f * Vec3f(0.5f, 1.1 - 0.13 * i, 0.12f + 0.21 * j + 0.1 * (8 - i));
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
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
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
	wireRender->setColor(Vec3f(0, 1, 0));
	contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	//Visualize contact points
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Vec3f(1, 0, 0));
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	return scn;
}

int main()
{
	QtApp window;
	window.setSceneGraph(createScene());
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}