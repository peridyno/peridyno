#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>

using namespace std;
using namespace dyno;

/**
 * This example demonstrates how to use the mouse to hit boxes
 */

__global__ void HitBoxes(
	DArray<TOrientedBox3D<float>> box,
	DArray<Vec3f> velocites,
	TRay3D<float> ray,
	int boxIndex)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId >= box.size()) return;

	TSegment3D<float> seg;

	if (ray.intersect(box[tId], seg) > 0)
	{
		velocites[tId + boxIndex] += 10.0f * ray.direction;
	};
}

class Hit : public MouseInputModule
{
public:
	Hit() {};
	virtual ~Hit() {};

	DEF_INSTANCE_IN(DiscreteElements<DataType3f>, Topology, "");

	DEF_ARRAY_IN(Vec3f, Velocity, DeviceType::GPU, "Rigid body velocities");

protected:
	void onEvent(PMouseEvent event) override {
		auto elements = this->inTopology()->getDataPtr();

		auto& velocities = this->inVelocity()->getData();

		ElementOffset offset = elements->calculateElementOffset();

		if (event.actionType == AT_PRESS)
		{
			cuExecute(velocities.size(),
				HitBoxes,
				elements->getBoxes(),
				velocities,
				event.ray,
				offset.boxIndex());
		}
	}
};

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
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(0.5f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	auto hit = std::make_shared<Hit>();
	rigid->stateTopology()->connect(hit->inTopology());
	rigid->stateVelocity()->connect(hit->inVelocity());
	rigid->animationPipeline()->pushModule(hit);

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


