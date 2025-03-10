#include "WtApp.h"

#include "SceneGraph.h"

#include "RigidBody/initializeRigidBody.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "DualParticleSystem/initializeDualParticleSystem.h"
#include "Peridynamics/initializePeridynamics.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "Volume/initializeVolume.h"
#include "Multiphysics/initializeMultiphysics.h"
#include "HeightField/initializeHeightField.h"
#include "initializeModeling.h"
#include "initializeIO.h"

#include "ObjIO/initializeObjIO.h"
#include "ObjIO/ObjLoader.h"



//#include <RigidBody/RigidBodySystem.h>
//#include <Module/MouseInputModule.h>
//#include <GLRenderEngine.h>
//#include <GLPointVisualModule.h>
//#include <GLSurfaceVisualModule.h>
//#include <GLWireframeVisualModule.h>
//
//#include <Mapping/DiscreteElementsToTriangleSet.h>
//using namespace std;
using namespace dyno;

//__global__ void HitBox(
//	DArray<TOrientedBox3D<float>> box,
//	DArray<Vec3f> velocites,
//	TRay3D<float> ray,
//	int boxIndex)
//{
//	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
//	if (tId >= box.size()) return;
//
//	TSegment3D<float> seg;
//
//	if (ray.intersect(box[tId], seg) > 0)
//	{
//		velocites[tId + boxIndex] += 10.0f * ray.direction;
//	};
//}
//
//class Hits : public MouseInputModule
//{
//public:
//	Hits() {};
//	virtual ~Hits() {};
//
//	DEF_INSTANCE_IN(DiscreteElements<DataType3f>, Topology, "");
//
//	DEF_ARRAY_IN(Vec3f, Velocity, DeviceType::GPU, "Rigid body velocities");
//
//protected:
//	void onEvent(PMouseEvent event) override {
//		auto elements = this->inTopology()->getDataPtr();
//
//		auto& velocities = this->inVelocity()->getData();
//
//		ElementOffset offset = elements->calculateElementOffset();
//
//		DArray<TOrientedBox3D<float>> boxInGlobal;
//
//		elements->requestBoxInGlobal(boxInGlobal);
//
//		if (event.actionType == AT_PRESS)
//		{
//			cuExecute(velocities.size(),
//				HitBox,
//				boxInGlobal,
//				velocities,
//				event.ray,
//				offset.boxIndex());
//		}
//
//		boxInGlobal.clear();
//	}
//};

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(1.5, 1, 1.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));
	return scn;

	/*std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.5, 0, 0);
	BoxInfo box;
	box.halfLength = 0.5f * Vec3f(0.065, 0.065, 0.1);
	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			rigidBody.position = 0.5f * Vec3f(0.5f, 1.1 - 0.13 * i, 0.12f + 0.21 * j + 0.1 * (8 - i));

			rigid->addBox(box, rigidBody);
		}

	SphereInfo sphere;
	sphere.radius = 0.025f;

	RigidBodyInfo rigidSphere;
	rigidSphere.position = Vec3f(0.5f, 0.75f, 0.5f);
	rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.5f, 0.95f, 0.5f);
	sphere.radius = 0.025f;
	rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.5f, 0.65f, 0.5f);
	sphere.radius = 0.05f;
	rigid->addSphere(sphere, rigidSphere);

	TetInfo tet;
	RigidBodyInfo rigidTet;
	tet.v[0] = Vec3f(0.5f, 1.1f, 0.5f);
	tet.v[1] = Vec3f(0.5f, 1.2f, 0.5f);
	tet.v[2] = Vec3f(0.6f, 1.1f, 0.5f);
	tet.v[3] = Vec3f(0.5f, 1.1f, 0.6f);
	rigid->addTet(tet, rigidTet);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(0.5f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	auto hit = std::make_shared<Hits>();
	rigid->stateTopology()->connect(hit->inTopology());
	rigid->stateVelocity()->connect(hit->inVelocity());
	rigid->animationPipeline()->pushModule(hit);

	return scn;*/
}

int main(int argc, char** argv)
{
	Modeling::initStaticPlugin();
	RigidBody::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	HeightFieldLibrary::initStaticPlugin();
	DualParticleSystem::initStaticPlugin();
	Peridynamics::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();
	Volume::initStaticPlugin();
	Multiphysics::initStaticPlugin();
	dynoIO::initStaticPlugin();
	ObjIO::initStaticPlugin();

	WtApp app;

	app.setSceneGraphCreator(&createScene);
	app.setSceneGraph(createScene());
	app.mainLoop();

	return 0;
}