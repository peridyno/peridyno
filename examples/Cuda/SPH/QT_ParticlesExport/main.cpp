#include <QtApp.h>
#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/GhostParticles.h>
#include <ParticleSystem/MakeGhostParticles.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h>
#include <ParticleSystem/GhostFluid.h>
#include <Samplers/ShapeSampler.h>
#include <Samplers/PointsBehindMesh.h>

#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>
#include <GLSurfaceVisualModule.h>

#include <BasicShapes/CubeModel.h>
#include <BasicShapes/PlaneModel.h>
#include <Volume/BasicShapeToVolume.h>
#include <Multiphysics/VolumeBoundary.h>

// IO
#include "PartioExporter/ParticleWriterBGEO.h"

using namespace std;
using namespace dyno;

bool useVTK = false;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(10.5, 5, 10.5));
	scn->setLowerBound(Vec3f(-10.5, -5, -10.5));

	auto plane = scn->addNode(std::make_shared <PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(0.5f, 0.0, 0.5f));

	/*Generate points on meshes, and relax the point positions*/
	auto solidPoints = scn->addNode(std::make_shared<ParticleRelaxtionOnMesh<DataType3f>>());
	solidPoints->varSamplingDistance()->setValue(0.005);
	solidPoints->varThickness()->setValue(0.045);
	solidPoints->varGeneratingDirection()->setValue(false);
	plane->stateTriangleSet()->connect(solidPoints->inTriangleSet());
	solidPoints->graphicsPipeline()->clear();

	auto ghostPoinsts = scn->addNode(std::make_shared<MakeGhostParticles<DataType3f>>());
	solidPoints->statePointSet()->connect(ghostPoinsts->inPoints());
	solidPoints->statePointNormal()->connect(ghostPoinsts->stateNormal());

	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0, 0.12, 0.0));
	cube->varLength()->setValue(Vec3f(0.2, 0.2, 0.2));
	cube->graphicsPipeline()->disable();

	auto sampler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->setVisible(false);

	cube->connect(sampler->importShape());

	auto fluidParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler->statePointSet()->promoteOuput()->connect(fluidParticles->inPoints());

	auto incompressibleFluid = scn->addNode(std::make_shared<GhostFluid<DataType3f>>());
	incompressibleFluid->varSmoothingLength()->setValue(2.5f);
	fluidParticles->connect(incompressibleFluid->importInitialStates());
	ghostPoinsts->connect(incompressibleFluid->importBoundaryParticles());

	//Create a container
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.0f, 0.5f, 0.0f));
	cubeBoundary->varLength()->setValue(Vec3f(0.4f, 1.0f, 0.4f));

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(boundary->importVolumes());
	incompressibleFluid->connect(boundary->importParticleSystems());
	
	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(0.6, 0.5, 0.2));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	solidPoints->statePointSet()->connect(ptRender->inPointSet());
	solidPoints->graphicsPipeline()->pushModule(ptRender);
	


    // BGEO
    auto bgeo = std::make_shared<ParticleWriterBGEO<DataType3f>>();
    incompressibleFluid->stateFrameNumber()->connect(bgeo->inFrameNumber());
    incompressibleFluid->statePosition()->connect(bgeo->inPosition());

    bgeo->varAttributeNum()->setValue(0);
	bgeo->varStartFrame()->setValue(0);
	bgeo->varEndFrame()->setValue(4320);
    bgeo->varStride()->setValue(4);
    bgeo->varOutputPath()->setValue(FilePath("H:\\Data\\Fluid\\Temp"));
    bgeo->varPrefix()->setValue("Particle_");

	incompressibleFluid->animationPipeline()->pushModule(bgeo);
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


