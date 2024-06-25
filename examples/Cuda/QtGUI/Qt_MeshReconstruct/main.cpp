#include <QtApp.h>
#include <SceneGraph.h>
#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/SquareEmitter.h>
#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <GLSurfaceVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>
#include "Node/GLPointVisualNode.h"
#include "Node/GLSurfaceVisualNode.h"
#include "ParticleSystem/StaticBoundary.h"

#include "DualParticleSystem/SurfaceLevelSetConstructionNode.h"
#include "Mapping\MarchingCubes.h"
#include "CubeModel.h"
#include "ParticleSystem/CubeSampler.h"
#include "ParticleSystem/MakeParticleSystem.h"


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(1.5, 1, 1.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));

	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.5, 0.1, 0.5));
	cube->varLength()->setValue(Vec3f(0.04, 0.04, 0.04));
	cube->setVisible(false);

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->setVisible(false);

	cube->outCube()->connect(sampler->inCube());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto emitter = scn->addNode(std::make_shared<SquareEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(0.5f));

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	initialParticles->connect(fluid->importInitialStates());
	emitter->connect(fluid->importParticleEmitters());

	std::shared_ptr<StaticBoundary<DataType3f>> root = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	root->loadCube(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(1.0f, 1.0f, 1.0f), 0.02, true);
	fluid->connect(root->importParticleSystems());

	auto meshRe = scn->addNode(std::make_shared<SurfaceLevelSetConstructionNode<DataType3f>>());
	meshRe->stateGridSpacing()->setValue(0.005);
	fluid->connect(meshRe->importParticleSystem());

	auto marchingCubes = scn->addNode(std::make_shared<MarchingCubes<DataType3f>>());
	meshRe->stateLevelSet()->connect(marchingCubes->inLevelSet());
	marchingCubes->varIsoValue()->setValue(-300000.0);
	marchingCubes->varGridSpacing()->setValue(0.005f);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(0.9, 0.9, 0.9));
	ptRender->varPointSize()->setValue(0.0015);
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	fluid->statePointSet()->connect(ptRender->inPointSet());
	fluid->graphicsPipeline()->pushModule(ptRender);

	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
	surfaceRenderer->setColor(Color(0.1f, 0.1f, 0.9f));
	marchingCubes->outTriangleSet()->connect(surfaceRenderer->inTriangleSet());
	surfaceRenderer->varAlpha()->setValue(0.3f);
	surfaceRenderer->varMetallic()->setValue(0.5f);
	marchingCubes->graphicsPipeline()->pushModule(surfaceRenderer);


	//auto surfaceRenderer = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	//surfaceRenderer->varColor()->setValue(Vec3f(16.0f, 75.0f, 204.0f));

	//marchingCubes->outTriangleSet()->connect(surfaceRenderer->inTriangleSet());
	//surfaceRenderer->setVisible(true);

	return scn;
}

int main()
{
	QtApp window;
	window.setSceneGraph(createScene());
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}