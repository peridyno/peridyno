#include <QtApp.h>
#include "Plugin/ObjIO/ObjLoader.h"
#include <QtApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/SquareEmitter.h>

#include <Multiphysics/SolidFluidCoupling.h>

#include <Module/CalculateNorm.h>
#include <Peridynamics/HyperelasticBody.h>
#include <Peridynamics/Cloth.h>
#include <Peridynamics/Thread.h>


#include "Node/GLPointVisualNode.h"
#include "Node/GLSurfaceVisualNode.h"

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>


#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"

#include "Topology/TriangleSet.h"
#include "Collision/NeighborPointQuery.h"

#include "ParticleWriter.h"
#include "EigenValueWriter.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>


#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"


#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/MakeParticleSystem.h"

#include "Topology/TriangleSet.h"
#include "Collision/NeighborPointQuery.h"

#include "ParticleWriter.h"
#include "EigenValueWriter.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLInstanceVisualModule.h>


#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <GLRenderEngine.h>

#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"

#include "StaticTriangularMesh.h"
#include "Plugin/ObjIO/ObjLoader.h"

#include "CubeModel.h"
#include "ParticleSystem/CubeSampler.h"
#include "SphereModel.h"
#include "Mapping/MergeTriangleSet.h"
#include "Merge.h"



using namespace dyno;


std::shared_ptr<SceneGraph> creatScene()
{	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！//
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-1.0f, 0.0f, 0.0f));
	scn->setUpperBound(Vec3f(1.0f, 1.0f, 1.0f));

	auto cube1 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube1->varLocation()->setValue(Vec3f(0.0, 0.025, 0.4));
	cube1->varLength()->setValue(Vec3f(0.35, 0.02 , 3 ));
	cube1->varScale()->setValue(Vec3f(2,1,1));
	cube1->graphicsPipeline()->disable();

	auto cubeSmapler1 = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	cubeSmapler1->varSamplingDistance()->setValue(0.005f);
	cube1->outCube()->connect(cubeSmapler1->inCube());
	cubeSmapler1->graphicsPipeline()->disable();

	auto outSamples = cubeSmapler1->statePointSet()->promoteOuput();

	auto makeParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	outSamples->connect(makeParticles->inPoints());

	//Particle fluid node
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	makeParticles->connect(fluid->importInitialStates());

	//fluid->loadParticles(Vec3f(-0.29, 0.005, 0.05), Vec3f(0.29, 0.05, 0.99), 0.005);

	//fluid->addFluidPointSet(cubeSmapler1);

	auto visualizer = scn->addNode(std::make_shared<GLPointVisualNode<DataType3f>>());

	auto outTop = fluid->statePointSet()->promoteOuput();
	auto outVel = fluid->stateVelocity()->promoteOuput();
	outTop->connect(visualizer->inPoints());
	outVel->connect(visualizer->inVector());


	fluid->animationPipeline()->disable();

	Vec3f velocity = Vec3f(0,0,6);
	Color color = Color(1, 1, 1);

	Vec3f LocationBody = Vec3f(0, 0.01, -1);
	Vec3f LocationLF = Vec3f(0.17,0.1,0.36) + LocationBody;
	Vec3f LocationLB = Vec3f(0.17, 0.1, -0.3) + LocationBody;
	Vec3f LocationRF = Vec3f(-0.17, 0.1, 0.36) + LocationBody;
	Vec3f LocationRB = Vec3f(-0.17, 0.1, -0.3) + LocationBody;

	Vec3f anglurVel = Vec3f(100,0,0);
	Vec3f scale = Vec3f(0.4,0.4,0.4);

	auto ObjJeep = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjJeep->varFileName()->setValue(getAssetPath() + "Jeep/jeep_low.obj");
	ObjJeep->varScale()->setValue(scale);
	ObjJeep->varLocation()->setValue(LocationBody);
	ObjJeep->varVelocity()->setValue(velocity);
	ObjJeep->surfacerender->setColor(color);
	

	auto ObjWheelLF = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjWheelLF->varFileName()->setValue(getAssetPath() + "Jeep/Wheel_R.obj");
	ObjWheelLF->varScale()->setValue(scale);
	ObjWheelLF->varLocation()->setValue(LocationBody);
	ObjWheelLF->varVelocity()->setValue(velocity);
	ObjWheelLF->surfacerender->setColor(color);
	ObjWheelLF->varLocation()->setValue(LocationLF);
	ObjWheelLF->varCenter()->setValue(LocationLF);
	ObjWheelLF->varAngularVelocity()->setValue(anglurVel);
	auto WheelLFTri = ObjWheelLF->outTriangleSet()->promoteOuput();


	auto ObjWheelLB = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjWheelLB->varFileName()->setValue(getAssetPath() + "Jeep/Wheel_R.obj");
	ObjWheelLB->varScale()->setValue(scale);
	ObjWheelLB->varLocation()->setValue(LocationBody);
	ObjWheelLB->varVelocity()->setValue(velocity);
	ObjWheelLB->surfacerender->setColor(color);
	ObjWheelLB->varLocation()->setValue(LocationLB);
	ObjWheelLB->varCenter()->setValue(LocationLB);
	ObjWheelLB->varAngularVelocity()->setValue(anglurVel);
	auto WheelLBTri = ObjWheelLB->outTriangleSet()->promoteOuput();


	auto ObjWheelRF = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjWheelRF->varFileName()->setValue(getAssetPath() + "Jeep/Wheel_L.obj");
	ObjWheelRF->varScale()->setValue(scale);
	ObjWheelRF->varLocation()->setValue(LocationBody);
	ObjWheelRF->varVelocity()->setValue(velocity);
	ObjWheelRF->surfacerender->setColor(color);
	ObjWheelRF->varLocation()->setValue(LocationRF);
	ObjWheelRF->varCenter()->setValue(LocationRF);
	ObjWheelRF->varAngularVelocity()->setValue(anglurVel);
	auto WheelRFTri = ObjWheelRF->outTriangleSet()->promoteOuput();


	auto ObjWheelRB = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjWheelRB->varFileName()->setValue(getAssetPath() + "Jeep/Wheel_R.obj");
	ObjWheelRB->varScale()->setValue(scale);
	ObjWheelRB->varLocation()->setValue(LocationBody);
	ObjWheelRB->varVelocity()->setValue(velocity);
	ObjWheelRB->surfacerender->setColor(color);
	ObjWheelRB->varLocation()->setValue(LocationRB);
	ObjWheelRB->varCenter()->setValue(LocationRB);
	ObjWheelRB->varAngularVelocity()->setValue(anglurVel);
	auto WheelRBTri = ObjWheelRB->outTriangleSet()->promoteOuput();


	//Road
	Vec3f LocationRoad = Vec3f(0,0,0.5);
	Vec3f ScaleRoad = Vec3f(0.04);

	auto ObjRoad = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoad->varFileName()->setValue(getAssetPath() + "Jeep/Road/Road.obj");
	ObjRoad->varScale()->setValue(ScaleRoad);
	ObjRoad->varLocation()->setValue(LocationRoad);
	ObjRoad->surfacerender->setColor(color);
	auto RoadTri = ObjRoad->outTriangleSet()->promoteOuput();


	auto ObjRoad_1 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoad_1->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_1.obj");
	ObjRoad_1->varScale()->setValue(ScaleRoad);
	ObjRoad_1->varLocation()->setValue(LocationRoad);
	ObjRoad_1->surfacerender->setColor(color);
	auto Road_1Tri = ObjRoad_1->outTriangleSet()->promoteOuput();


	auto ObjRoadWall = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoadWall->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_wall.obj");
	ObjRoadWall->varScale()->setValue(ScaleRoad);
	ObjRoadWall->varLocation()->setValue(LocationRoad);
	ObjRoadWall->surfacerender->setColor(color);
	auto RoadWallTri = ObjRoadWall->outTriangleSet()->promoteOuput();
	

	auto ObjRoadDoor = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoadDoor->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_door.obj");
	ObjRoadDoor->varScale()->setValue(ScaleRoad);
	ObjRoadDoor->varLocation()->setValue(LocationRoad);
	ObjRoadDoor->surfacerender->setColor(Color(0.5));
	ObjRoadDoor->surfacerender->setRoughness(0.5);
	ObjRoadDoor->surfacerender->setMetallic(1);



	auto ObjRoadLogo = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoadLogo->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_logo.obj");
	ObjRoadLogo->varScale()->setValue(ScaleRoad);
	ObjRoadLogo->varLocation()->setValue(LocationRoad);
	ObjRoadLogo->surfacerender->setColor(Color(0,0.2,1));

	auto ObjRoadText = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoadText->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_peridyno.obj");
	ObjRoadText->varScale()->setValue(ScaleRoad);
	ObjRoadText->varLocation()->setValue(LocationRoad);
	ObjRoadText->surfacerender->setColor(Color(4,4,4));

	//Scene boundary
	auto boundary = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	boundary->varFileName()->setValue(getAssetPath() + "Jeep/Road/boundary.obj");
	boundary->varScale()->setValue(ScaleRoad);
	boundary->varLocation()->setValue(LocationRoad);

	auto SurfaceModule1 = boundary->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule1->setVisible(0);


	//SFI node
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalSFINode<DataType3f>>());
	sfi->varFast()->setValue(true);

	fluid->connect(sfi->importParticleSystems());


	auto merge1 = scn->addNode(std::make_shared<Merge<DataType3f>>());
	merge1->varUpdateMode()->setCurrentKey(1);

	ObjWheelLF->outTriangleSet()->connect(merge1->inTriangleSet01());
	ObjWheelLB->outTriangleSet()->connect(merge1->inTriangleSet02());
	ObjWheelRF->outTriangleSet()->connect(merge1->inTriangleSet03());
	ObjWheelRB->outTriangleSet()->connect(merge1->inTriangleSet04());

	auto merge2 = scn->addNode(std::make_shared<Merge<DataType3f>>());
	merge2->varUpdateMode()->setCurrentKey(1);
	merge1->stateTriangleSet()->promoteOuput()->connect(merge2->inTriangleSet01());
	ObjRoad->outTriangleSet()->promoteOuput()->connect(merge2->inTriangleSet02());

	merge2->stateTriangleSet()->promoteOuput()->connect(sfi->inTriangleSet());

	return scn;
}

int main()
{
	QtApp window;
	window.setSceneGraph(creatScene());
	window.initialize(1366, 768);
	window.mainLoop();

	return 0;
}