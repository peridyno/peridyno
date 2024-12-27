#include <UbiApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>
#include <RigidBody/MultibodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include <BasicShapes/PlaneModel.h>

#include <Collision/NeighborElementQuery.h>

using namespace std;
using namespace dyno;

std::shared_ptr<RigidBodySystem<DataType3f>> createCompound(std::shared_ptr<SceneGraph> scn)
{
	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	auto actor = rigid->createRigidBody(Vec3f(0.0f, 1.3f, 0.0f), Quat<float>(0.5, Vec3f(1.0f, 0.0f, 1.0f)));

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

	auto actor2 = rigid->createRigidBody(Vec3f(-0.1f, 1.6f, 0.0f), Quat<float>());
	SphereInfo sphere2;
	sphere2.center = Vec3f(0.0f, 0.0f, 0.0f);
	sphere2.radius = 0.1f;
	rigid->bindSphere(actor2, sphere2);

// 	auto& hingeJoint = rigid->createHingeJoint(actor, actor2);
// 	hingeJoint.setAnchorPoint((Vec3f(0.0f, 1.3f, 0.0f) + Vec3f(-0.1f, 1.6f, 0.0f)) / 2);
// 	hingeJoint.setAxis(Vec3f(0, 0, 1));
// 	hingeJoint.setRange(-M_PI / 2, M_PI / 2);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(0.5f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	return rigid;
}

std::shared_ptr<RigidBodySystem<DataType3f>> createBoxes(std::shared_ptr<SceneGraph> scn)
{
	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());
	uint dim = 2;
	float h = 0.1f;

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	BoxInfo box;
	box.halfLength = Vec3f(h, h, h);
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			for (int k = 0; k < dim; k++)
			{
				rigidBody.position = Vec3f(2 * i * h - h * dim, h + (2.01f) * j * h, 2 * k * h - h * dim);

				auto boxAt = rigid->addBox(box, rigidBody);
			}
		}
	}

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color::SteelBlue2());
	sRender->setAlpha(1.0f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Color(0, 0, 0));
	mapper->outTriangleSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	return rigid;
}

std::shared_ptr<SceneGraph> createSceneGraph()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto compound = createCompound(scn);
	auto boxes = createBoxes(scn);

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(5);
	plane->varLengthZ()->setValue(5);

	auto convoy = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	boxes->connect(convoy->importVehicles());
	compound->connect(convoy->importVehicles());
	plane->stateTriangleSet()->connect(convoy->inTriangleSet());

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createSceneGraph());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}