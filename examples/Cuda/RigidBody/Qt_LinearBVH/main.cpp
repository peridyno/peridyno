#include <QtApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/BoundingBoxToEdgeSet.h>

#include "Collision/NeighborElementQuery.h"
#include "Collision/CalculateBoundingBox.h"

#include <Topology/LinearBVH.h>

using namespace std;
using namespace dyno;

/**
 * This example demonstrates how to construct linear BVH given a set of AABBs
 */


template<typename TDataType>
class ConstructLinearBVH : public ComputeModule
{	
	DECLARE_TCLASS(ConstructLinearBVH, TDataType)

	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename ::dyno::TAlignedBox3D<Real> AABB;

public:
	ConstructLinearBVH() {};
	~ConstructLinearBVH() override { bvh.release(); }

	void compute() override
	{
		auto& inAABB = this->inAABB()->getData();

		bvh.construct(inAABB);

		this->outAABB()->assign(bvh.getSortedAABBs());
	};

	DEF_ARRAY_IN(AABB, AABB, DeviceType::GPU, "");

	DEF_ARRAY_OUT(AABB, AABB, DeviceType::GPU, "");

private:
	LinearBVH<TDataType> bvh;
};

IMPLEMENT_TCLASS(ConstructLinearBVH, TDataType)

std::shared_ptr<SceneGraph> createBoxes()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	BoxInfo box;
	box.center = Vec3f(0.0f);
	box.halfLength = Vec3f(0.1, 0.1, 0.1);

	RigidBodyInfo rigidBody;
	rigidBody.position = Vec3f(-0.3, 0.1, 0.5);
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	rigid->addBox(box, rigidBody);

	rigidBody.position = Vec3f(-0.3, 0.3, 0.59);
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	rigid->addBox(box, rigidBody);

	SphereInfo sphere;
	sphere.radius = 0.025f;

	RigidBodyInfo rigidSphere;
	rigidSphere.position = Vec3f(0.5f, 0.75f, 0.5f);
	rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.7f, 0.95f, 0.5f);
	sphere.radius = 0.025f;
	rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.3f, 0.65f, 0.5f);
	sphere.radius = 0.05f;
	rigid->addSphere(sphere, rigidSphere);

	TetInfo tet;
	tet.v[0] = Vec3f(0.5f, 1.1f, 0.5f);
	tet.v[1] = Vec3f(0.5f, 1.2f, 0.5f);
	tet.v[2] = Vec3f(0.6f, 1.1f, 0.5f);
	tet.v[3] = Vec3f(0.5f, 1.1f, 0.6f);

	RigidBodyInfo tetRigid;
	rigid->addTet(tet, tetRigid);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(0.5f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	//Visualize bounding boxes
	auto computeAABB = std::make_shared<CalculateBoundingBox<DataType3f>>();
	rigid->stateTopology()->connect(computeAABB->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(computeAABB);

	auto bvhConstructor = std::make_shared<ConstructLinearBVH<DataType3f>>();
	computeAABB->outAABB()->connect(bvhConstructor->inAABB());
	rigid->graphicsPipeline()->pushModule(bvhConstructor);

	auto bvhMapper = std::make_shared<BoundingBoxToEdgeSet<DataType3f>>();
	bvhConstructor->outAABB()->connect(bvhMapper->inAABB());
	rigid->graphicsPipeline()->pushModule(bvhMapper);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Color(0, 0, 1));
	bvhMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(createBoxes());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


