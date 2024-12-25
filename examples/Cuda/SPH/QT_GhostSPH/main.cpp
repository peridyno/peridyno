//#include <GlfwApp.h>
#include <QtApp.h>
#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include "ParticleSystem/GhostParticles.h"
#include "ParticleSystem/MakeGhostParticles.h"
#include <ParticleSystem/GhostFluid.h>
#include <Samplers/CubeSampler.h>
#include "ParticleSystem/MakeParticleSystem.h"

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <ColorMapping.h>
#include <ImColorbar.h>

#include <BasicShapes/CubeModel.h>
#include <BasicShapes/SphereModel.h>
#include <BasicShapes/PlaneModel.h>
#include "Samplers/PointsBehindMesh.h"
#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"
#include "ObjIO/ObjLoader.h"
#include "GLSurfaceVisualModule.h"

using namespace std;
using namespace dyno;

bool useVTK = false;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(10.5, 5, 10.5));
	scn->setLowerBound(Vec3f(-10.5, -5, -10.5));

	auto obj1 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	obj1->varScale()->setValue(Vec3f(0.3));
	obj1->varFileName()->setValue(getAssetPath() + "plane/plane_lowRes.obj");
	//obj1->varFileName()->setValue(getAssetPath() + "board/ball.obj");
	obj1->varLocation()->setValue(Vec3f(0.0, 0.0, 0.0));
	auto SurfaceModule1 = obj1->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule1->setColor(Color(0.2));
	SurfaceModule1->setMetallic(1);
	SurfaceModule1->setRoughness(0.8);

	//auto pointset_1 = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	//pointset_1->varGeneratingDirection()->setValue(false);
	//pointset_1->varSamplingDistance()->setValue(0.005);
	//pointset_1->varThickness()->setValue(0.045);
	//obj1->outTriangleSet()->connect(pointset_1->inTriangleSet());

	/*Generate points on meshes of the cube, and relax the point positions*/
	auto pointset_1 = scn->addNode(std::make_shared<ParticleRelaxtionOnMesh<DataType3f>>());
	pointset_1->varSamplingDistance()->setValue(0.005);
	pointset_1->varThickness()->setValue(0.045);
	obj1->outTriangleSet()->connect(pointset_1->inTriangleSet());
	pointset_1->graphicsPipeline()->clear();

	//auto pointset_1 = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	//pointset_1->varSamplingDistance()->setValue(0.005);
	//pointset_1->varThickness()->setValue(0.045);
	//obj1->outTriangleSet()->connect(pointset_1->inTriangleSet());
	//pointset_1->graphicsPipeline()->clear();

	auto ghost2 = scn->addNode(std::make_shared<MakeGhostParticles<DataType3f>>());
	pointset_1->statePointSet()->connect(ghost2->inPoints());
	pointset_1->statePointNormal()->connect(ghost2->stateNormal());

//	auto ghost2 = scn->addNode(createGhostParticles());


	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0, 0.3, 0.0));
	cube->varLength()->setValue(Vec3f(0.2, 0.2, 0.2));
	cube->graphicsPipeline()->disable();

	
	auto sampler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->setVisible(false);

	cube->outCube()->connect(sampler->inCube());

	auto fluidParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());

	sampler->statePointSet()->promoteOuput()->connect(fluidParticles->inPoints());

	auto incompressibleFluid = scn->addNode(std::make_shared<GhostFluid<DataType3f>>());
	fluidParticles->connect(incompressibleFluid->importInitialStates());
	ghost2->connect(incompressibleFluid->importBoundaryParticles());

	{
		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(0.6, 0.5, 0.2));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		pointset_1->statePointSet()->connect(ptRender->inPointSet());
		pointset_1->graphicsPipeline()->pushModule(ptRender);
	}
	return scn;
}

int main()
{
	//GlfwApp app;
	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


