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
#include "ColorMapping.h"



using namespace dyno;

std::shared_ptr<SceneGraph> creatScene();
void importOtherModel(std::shared_ptr<SceneGraph> scn);

std::shared_ptr<SceneGraph> creatScene()
{	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	//***************************************Scene Setting***************************************//
	// Scene Setting
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-1.0f, 0.0f, 0.0f));
	scn->setUpperBound(Vec3f(1.0f, 1.0f, 1.0f));


	// Create Var
	Vec3f velocity = Vec3f(0,0,6);
	Color color = Color(1, 1, 1);

	Vec3f LocationBody = Vec3f(0, 0.01, -1);

	Vec3f anglurVel = Vec3f(100,0,0); 
	Vec3f scale = Vec3f(0.4,0.4,0.4);



	//*************************************** Import Model ***************************************//
	// Import Jeep
	auto ObjJeep = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjJeep->varFileName()->setValue(getAssetPath() + "Jeep/jeep_low.obj");
	ObjJeep->varScale()->setValue(scale);
	ObjJeep->varLocation()->setValue(LocationBody);
	ObjJeep->varVelocity()->setValue(velocity);
	ObjJeep->surfacerender->setColor(color);

	// Import Wheel
	std::vector<std::string> wheelPath = { "Jeep/Wheel_R.obj","Jeep/Wheel_R.obj","Jeep/Wheel_L.obj","Jeep/Wheel_R.obj" };
	std::vector<std::shared_ptr<ObjMesh<DataType3f>>> wheelSet;

	std::vector<Vec3f> wheelLocation;
	wheelLocation.push_back(Vec3f(0.17, 0.1, 0.36) + LocationBody);
	wheelLocation.push_back(Vec3f(0.17, 0.1, -0.3) + LocationBody);
	wheelLocation.push_back(Vec3f(-0.17, 0.1, 0.36) + LocationBody);
	wheelLocation.push_back(Vec3f(-0.17, 0.1, -0.3) + LocationBody);

	for (int i = 0; i < 4; i++) 
	{
		auto ObjWheel = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
		ObjWheel->varFileName()->setValue(getAssetPath() + wheelPath[i]);

		ObjWheel->varScale()->setValue(scale);
		ObjWheel->varLocation()->setValue(wheelLocation[i]);
		ObjWheel->varCenter()->setValue(wheelLocation[i]);

		ObjWheel->varVelocity()->setValue(velocity);
		ObjWheel->varAngularVelocity()->setValue(anglurVel);

		wheelSet.push_back(ObjWheel);
	}

	// Import Road

	auto ObjRoad = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoad->varFileName()->setValue(getAssetPath() + "Jeep/Road/Road.obj");
	ObjRoad->varScale()->setValue(Vec3f(0.04));
	ObjRoad->varLocation()->setValue(Vec3f(0, 0, 0.5));
	ObjRoad->surfacerender->setColor(Color(1, 1, 1));

	//*************************************** Merge Model ***************************************//
	//MergeWheel
	auto mergeWheel = scn->addNode(std::make_shared<Merge<DataType3f>>());
	mergeWheel->varUpdateMode()->setCurrentKey(1);

	wheelSet[0]->outTriangleSet()->connect(mergeWheel->inTriangleSet01());
	wheelSet[1]->outTriangleSet()->connect(mergeWheel->inTriangleSet02());
	wheelSet[2]->outTriangleSet()->connect(mergeWheel->inTriangleSet03());
	wheelSet[3]->outTriangleSet()->connect(mergeWheel->inTriangleSet04());

	//MergeRoad
	auto mergeRoad = scn->addNode(std::make_shared<Merge<DataType3f>>());
	mergeRoad->varUpdateMode()->setCurrentKey(1);
	mergeWheel->stateTriangleSet()->promoteOuput()->connect(mergeRoad->inTriangleSet01());
	ObjRoad->outTriangleSet()->connect(mergeRoad->inTriangleSet02());

	//Obj boundary
	auto ObjBoundary = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjBoundary->varFileName()->setValue(getAssetPath() + "Jeep/Road/boundary.obj");
	ObjBoundary->varScale()->setValue(Vec3f(0.04));
	ObjBoundary->varLocation()->setValue(Vec3f(0, 0, 0.5));
	ObjBoundary->surfacerender->setColor(Color(1, 1, 1));

	ObjBoundary->outTriangleSet()->connect(mergeRoad->inTriangleSet02());
	ObjBoundary->graphicsPipeline()->disable();

	//SetVisible
	mergeRoad->graphicsPipeline()->disable();

	//*************************************** Cube Sample ***************************************//
	// Cube 
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0, 0.025, 0.4));
	cube->varLength()->setValue(Vec3f(0.35, 0.02, 3));
	cube->varScale()->setValue(Vec3f(2, 1, 1));
	cube->graphicsPipeline()->disable();

	auto cubeSmapler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	cubeSmapler->varSamplingDistance()->setValue(0.005f);
	cube->outCube()->connect(cubeSmapler->inCube());
	cubeSmapler->graphicsPipeline()->disable();

	//MakeParticleSystem
	auto particleSystem = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	cubeSmapler->statePointSet()->promoteOuput()->connect(particleSystem->inPoints());


	//*************************************** Fluid ***************************************//
	//Particle fluid node
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	particleSystem->connect(fluid->importInitialStates());

	auto visualizer = scn->addNode(std::make_shared<GLPointVisualNode<DataType3f>>());

	fluid->statePointSet()->promoteOuput()->connect(visualizer->inPoints());
	fluid->stateVelocity()->promoteOuput()->connect(visualizer->inVector());

	//SemiAnalyticalSFINode
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalSFINode<DataType3f>>());
	sfi->varFast()->setValue(true);
	fluid->connect(sfi->importParticleSystems());
	mergeRoad->stateTriangleSet()->promoteOuput()->connect(sfi->inTriangleSet());

	//Create a boundary
	auto staticBoundary = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>()); ;
	staticBoundary->loadCube(Vec3f(-1.0, 0, -1.5), Vec3f(1.0, 2, 3.0), 0.02, true);
	fluid->connect(staticBoundary->importParticleSystems());

	//firstModule
	auto colormapping = visualizer->graphicsPipeline()->findFirstModule<ColorMapping<DataType3f>>();
	colormapping->varMax()->setValue(1.5);

	//*************************************** Import Other Models ***************************************//

	//Other Models
	importOtherModel(scn);

	return scn;
}


void importOtherModel(std::shared_ptr<SceneGraph> scn)
{
	//Other Models
	Vec3f LocationRoad = Vec3f(0, 0, 0.5);
	Vec3f ScaleRoad = Vec3f(0.04);

	auto ObjRoad_1 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoad_1->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_1.obj");
	ObjRoad_1->varScale()->setValue(ScaleRoad);
	ObjRoad_1->varLocation()->setValue(LocationRoad);
	ObjRoad_1->surfacerender->setColor(Color(1, 1, 1));

	auto ObjRoadWall = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoadWall->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_wall.obj");
	ObjRoadWall->varScale()->setValue(ScaleRoad);
	ObjRoadWall->varLocation()->setValue(LocationRoad);
	ObjRoadWall->surfacerender->setColor(Color(1, 1, 1));

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
	ObjRoadLogo->surfacerender->setColor(Color(0, 0.2, 1));

	auto ObjRoadText = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	ObjRoadText->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_peridyno.obj");
	ObjRoadText->varScale()->setValue(ScaleRoad);
	ObjRoadText->varLocation()->setValue(LocationRoad);
	ObjRoadText->surfacerender->setColor(Color(4, 4, 4));

}

int main()
{
	QtApp window;
	window.setSceneGraph(creatScene());
	window.initialize(1366, 768);
	window.mainLoop();

	return 0;
}