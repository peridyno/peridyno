#include <UbiApp.h>

#include "SceneGraph.h"
#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <ParticleSystem/Emitters/SquareEmitter.h>

#include <GLSurfaceVisualModule.h>

#include <Commands/Merge.h>

#include <BasicShapes/CubeModel.h>
#include <Samplers/ShapeSampler.h>

#include <Node/GLPointVisualNode.h>

#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>

#include <ColorMapping.h>

#include "Auxiliary/DataSource.h"

#include <ObjIO/ObjLoader.h>
#include "GLPointVisualModule.h"

using namespace dyno;
std::shared_ptr<SceneGraph> creatScene();
void importOtherModel(std::shared_ptr<SceneGraph> scn);


std::shared_ptr<SceneGraph> creatScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	//***************************************Scene Setting***************************************//
	// Scene Setting
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-0.5f, 0.0f, -4.0f));
	scn->setUpperBound(Vec3f(0.5f, 1.0f, 4.0f));


	// Create Var
	Vec3f velocity = Vec3f(0, 0, 6);
	Color color = Color(1, 1, 1);

	Vec3f LocationBody = Vec3f(0, 0.01, -1);

	Vec3f anglurVel = Vec3f(100, 0, 0);
	Vec3f scale = Vec3f(0.4, 0.4, 0.4);



	//*************************************** Import Model ***************************************//
	// Import Jeep
	auto ObjJeep = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjJeep->varFileName()->setValue(getAssetPath() + "Jeep/jeep_low.obj");
	ObjJeep->varScale()->setValue(scale);
	ObjJeep->varLocation()->setValue(LocationBody);
	ObjJeep->varVelocity()->setValue(velocity);
	auto glJeep = ObjJeep->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glJeep->setColor(color);

	// Import Wheel
	std::vector<std::string> wheelPath = { "Jeep/Wheel_R.obj","Jeep/Wheel_R.obj","Jeep/Wheel_L.obj","Jeep/Wheel_R.obj" };
	std::vector<std::shared_ptr<ObjLoader<DataType3f>>> wheelSet;

	std::vector<Vec3f> wheelLocation;
	wheelLocation.push_back(Vec3f(0.17, 0.1, 0.36) + LocationBody);
	wheelLocation.push_back(Vec3f(0.17, 0.1, -0.3) + LocationBody);
	wheelLocation.push_back(Vec3f(-0.17, 0.1, 0.36) + LocationBody);
	wheelLocation.push_back(Vec3f(-0.17, 0.1, -0.3) + LocationBody);

	for (int i = 0; i < 4; i++)
	{
		auto ObjWheel = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
		ObjWheel->varFileName()->setValue(getAssetPath() + wheelPath[i]);

		ObjWheel->varScale()->setValue(scale);
		ObjWheel->varLocation()->setValue(wheelLocation[i]);
		ObjWheel->varCenter()->setValue(wheelLocation[i]);

		ObjWheel->varVelocity()->setValue(velocity);
		ObjWheel->varAngularVelocity()->setValue(anglurVel);

		wheelSet.push_back(ObjWheel);
	}

	// Import Road

	auto ObjRoad = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjRoad->varFileName()->setValue(getAssetPath() + "Jeep/Road/Road.obj");
	ObjRoad->varScale()->setValue(Vec3f(0.04));
	ObjRoad->varLocation()->setValue(Vec3f(0, 0, 0.5));
	auto glRoad = ObjRoad->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glRoad->setColor(color);

	//*************************************** Merge Model ***************************************//
	//MergeWheel
	auto mergeWheel = scn->addNode(std::make_shared<Merge<DataType3f>>());
	mergeWheel->varUpdateMode()->setCurrentKey(1);

	wheelSet[0]->outTriangleSet()->connect(mergeWheel->inTriangleSets());
	wheelSet[1]->outTriangleSet()->connect(mergeWheel->inTriangleSets());
	wheelSet[2]->outTriangleSet()->connect(mergeWheel->inTriangleSets());
	wheelSet[3]->outTriangleSet()->connect(mergeWheel->inTriangleSets());

	//MergeRoad
	auto mergeRoad = scn->addNode(std::make_shared<Merge<DataType3f>>());
	mergeRoad->varUpdateMode()->setCurrentKey(1);
	mergeWheel->stateTriangleSet()->promoteOuput()->connect(mergeRoad->inTriangleSets());
	ObjRoad->outTriangleSet()->connect(mergeRoad->inTriangleSets());

	//Obj boundary
	auto ObjBoundary = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjBoundary->varFileName()->setValue(getAssetPath() + "Jeep/Road/boundary.obj");
	ObjBoundary->varScale()->setValue(Vec3f(0.04));
	ObjBoundary->varLocation()->setValue(Vec3f(0, 0, 0.5));
	auto glBoundary = ObjBoundary->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glBoundary->setColor(color);

	ObjBoundary->outTriangleSet()->connect(mergeRoad->inTriangleSets());
	ObjBoundary->graphicsPipeline()->disable();
	ObjJeep->outTriangleSet()->connect(mergeRoad->inTriangleSets());

	//SetVisible
	mergeRoad->graphicsPipeline()->disable();

	//*************************************** Cube Sample ***************************************//
	// Cube 
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0, 0.025, 0.4));
	cube->varLength()->setValue(Vec3f(0.35, 0.02, 3));
	cube->varScale()->setValue(Vec3f(2, 1, 1));
	cube->graphicsPipeline()->disable();

	auto cubeSmapler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	cubeSmapler->varSamplingDistance()->setValue(0.005f);
	cube->connect(cubeSmapler->importShape());
	cubeSmapler->graphicsPipeline()->disable();

	//MakeParticleSystem
	auto particleSystem = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	cubeSmapler->statePointSet()->promoteOuput()->connect(particleSystem->inPoints());


	//*************************************** Fluid ***************************************//
	//Particle fluid node
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	particleSystem->connect(fluid->importInitialStates());

	auto visualizer = scn->addNode(std::make_shared<GLPointVisualNode<DataType3f>>());
	auto ptrender = visualizer->graphicsPipeline()->findFirstModule<GLPointVisualModule>();
	ptrender->varPointSize()->setValue(0.001);

	fluid->statePointSet()->promoteOuput()->connect(visualizer->inPoints());
	fluid->stateVelocity()->promoteOuput()->connect(visualizer->inVector());

	//SemiAnalyticalSFINode
	auto meshBoundary = scn->addNode(std::make_shared<TriangularMeshBoundary<DataType3f>>());
	//sfi->varFast()->setValue(true);
	fluid->connect(meshBoundary->importParticleSystems());

	mergeRoad->stateTriangleSet()->promoteOuput()->connect(meshBoundary->inTriangleSet());

	//Create a boundary
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.0f, 1.0f, 0.75f));
	cubeBoundary->varLength()->setValue(Vec3f(2.0f, 2.0f, 4.5f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(container->importVolumes());

	fluid->connect(container->importParticleSystems());

	//firstModule
	auto colormapping = visualizer->graphicsPipeline()->findFirstModule<ColorMapping<DataType3f>>();
	colormapping->varMax()->setValue(1.5);

	//*************************************** Import Other Models ***************************************//

	//Other Models
	//importOtherModel(scn);

	return scn;

}

void importOtherModel(std::shared_ptr<SceneGraph> scn)
{
	//Other Models
	Vec3f LocationRoad = Vec3f(0, 0, 0.5);
	Vec3f ScaleRoad = Vec3f(0.04);

	auto ObjRoad_1 = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjRoad_1->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_1.obj");
	ObjRoad_1->varScale()->setValue(ScaleRoad);
	ObjRoad_1->varLocation()->setValue(LocationRoad);
	auto glRoad_1 = ObjRoad_1->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glRoad_1->setColor(Color(1, 1, 1));

	auto ObjRoadWall = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjRoadWall->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_wall.obj");
	ObjRoadWall->varScale()->setValue(ScaleRoad);
	ObjRoadWall->varLocation()->setValue(LocationRoad);
	auto glRoadWall = ObjRoadWall->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glRoadWall->setColor(Color(1, 1, 1));

	auto ObjRoadDoor = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjRoadDoor->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_door.obj");
	ObjRoadDoor->varScale()->setValue(ScaleRoad);
	ObjRoadDoor->varLocation()->setValue(LocationRoad);
	auto glRoadDoor = ObjRoadDoor->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glRoadDoor->setColor(Color(0.5));
	glRoadDoor->setRoughness(0.5);
	glRoadDoor->setMetallic(1);

	auto ObjRoadLogo = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjRoadLogo->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_logo.obj");
	ObjRoadLogo->varScale()->setValue(ScaleRoad);
	ObjRoadLogo->varLocation()->setValue(LocationRoad);
	auto glRoadLogo = ObjRoadLogo->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glRoadLogo->setColor(Color(0, 0.2, 1));

	auto ObjRoadText = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjRoadText->varFileName()->setValue(getAssetPath() + "Jeep/Road/obj_peridyno.obj");
	ObjRoadText->varScale()->setValue(ScaleRoad);
	ObjRoadText->varLocation()->setValue(LocationRoad);
	auto glRoadText = ObjRoadText->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glRoadText->setColor(Color(4, 4, 4));

}

int main(int argc, char** argv)
{
	UbiApp app(GUIType::GUI_WT);
	app.setSceneGraphCreator(&creatScene);
	app.setSceneGraph(creatScene());
	//app.initialize(1024, 768);
	app.mainLoop();
	return 0;
}