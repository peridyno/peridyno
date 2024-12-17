#include <UbiApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Topology/PointSet.h>

#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <Peridynamics/ElastoplasticBody.h>
#include <Peridynamics/ElasticBody.h>


#include <RigidBody/RigidBody.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>
#include <GLPointVisualModule.h>

#include <Node/GLSurfaceVisualNode.h>

#include <Mapping/PointSetToTriangleSet.h>

#include <SurfaceMeshLoader.h>

// Modeling
#include <BasicShapes/CubeModel.h>

// ParticleSystem
#include <ParticleSystem/ParticleFluid.h>
#include "ParticleSystem/MakeParticleSystem.h"

#include <Samplers/CubeSampler.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	//Create a scene graph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.2, 0.2, 0.0));
	cube->varLength()->setValue(Vec3f(0.1, 0.1, 0.1));
	cube->varSegments()->setValue(Vec3i(10, 10, 10));

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->graphicsPipeline()->disable();

	cube->outCube()->connect(sampler->inCube());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());

	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	//Create a elastoplastic object
	auto elastoplasticBody = scn->addNode(std::make_shared<ElastoplasticBody<DataType3f>>());	
	initialParticles->connect(elastoplasticBody->importSolidParticles());

	auto elastoplasticBodyRenderer = std::make_shared<GLPointVisualModule>();
	elastoplasticBodyRenderer->varPointSize()->setValue(0.005);
	elastoplasticBodyRenderer->setColor(Color(1, 0.2, 1));
	elastoplasticBodyRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
	elastoplasticBody->statePointSet()->connect(elastoplasticBodyRenderer->inPointSet());
	elastoplasticBody->stateVelocity()->connect(elastoplasticBodyRenderer->inColor());
	elastoplasticBody->graphicsPipeline()->pushModule(elastoplasticBodyRenderer);

// 	//Create a surface mesh loader
// 	auto surfaceMeshLoader = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
// 	surfaceMeshLoader->varFileName()->setValue(getAssetPath() + "standard/standard_cube20.obj");
// 	surfaceMeshLoader->varScale()->setValue(Vec3f(0.05f));
// 	surfaceMeshLoader->varLocation()->setValue(Vec3f(0.2, 0.2, 0.0));

	//Create a topology mapper
	auto topoMapper = scn->addNode(std::make_shared<PointSetToTriangleSet<DataType3f>>());

	auto outTop = elastoplasticBody->statePointSet()->promoteOuput();
 	outTop->connect(topoMapper->inPointSet());
	cube->stateTriangleSet()->connect(topoMapper->inInitialShape());

	auto surfaceVisualizer = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	topoMapper->outShape()->connect(surfaceVisualizer->inTriangleSet());

/*	//Create a cube 2
	auto cube2 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube2->varLocation()->setValue(Vec3f(-0.2, 0.2, 0.0));
	cube2->varLength()->setValue(Vec3f(0.1, 0.1, 0.1));
	cube2->varSegments()->setValue(Vec3i(10, 10, 10));

	//Create a sampler 2
	auto sampler2 = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	sampler2->varSamplingDistance()->setValue(0.005);
	sampler2->graphicsPipeline()->disable();

	cube2->outCube()->connect(sampler2->inCube());

	auto initialParticles2 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler2->statePointSet()->promoteOuput()->connect(initialParticles2->inPoints());

	//Create a elastic object
	auto elasticBody = scn->addNode(std::make_shared<ElasticBody<DataType3f>>());
	initialParticles2->connect(elasticBody->importSolidParticles());

	auto elasticBodyRenderer = std::make_shared<GLPointVisualModule>();
	elasticBodyRenderer->varPointSize()->setValue(0.005);
	elasticBodyRenderer->setColor(Color(1, 0.2, 1));
	elasticBodyRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
	elasticBody->statePointSet()->connect(elasticBodyRenderer->inPointSet());
	elasticBody->stateVelocity()->connect(elasticBodyRenderer->inColor());
	elasticBody->graphicsPipeline()->pushModule(elasticBodyRenderer);

	//Create a topology mapper
	auto topoMapper2 = scn->addNode(std::make_shared<PointSetToTriangleSet<DataType3f>>());

	auto outTop2 = elasticBody->statePointSet()->promoteOuput();
	outTop2->connect(topoMapper2->inPointSet());
	cube2->stateTriangleSet()->connect(topoMapper2->inInitialShape());

	auto surfaceVisualizer2 = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	topoMapper2->outShape()->connect(surfaceVisualizer2->inTriangleSet());*/

	//Create a container
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.5f, 1.0f, 0.5f));
	cubeBoundary->varLength()->setValue(Vec3f(2.0f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(container->importVolumes());

	elastoplasticBody->connect(container->importParticleSystems());
	//elasticBody->connect(container->importParticleSystems());

	return scn;
}

int main(int argc, char* argv[])
{
	UbiApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


