#include <GlfwApp.h>
#include <SceneGraph.h>

#include <Volume/BasicShapeToVolume.h>
#include <Multiphysics/VolumeBoundary.h>

#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>

#include "DualParticleSystem/DualParticleFluid.h"
#include "ParticleSystem/MakeParticleSystem.h"
#include <BasicShapes/CubeModel.h>
#include <Samplers/ShapeSampler.h>
#include <ParticleSystem/Emitters/SquareEmitter.h>

using namespace std;
using namespace dyno;

bool useVTK = false;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(3.0, 3.0, 3.0));
	scn->setLowerBound(Vec3f(-3.0, -3.0, -3.0));

	auto cube1 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube1->varLocation()->setValue(Vec3f(0.125, 0.125, 0.125));
	cube1->varLength()->setValue(Vec3f(0.15, 0.15, 0.15));
	cube1->graphicsPipeline()->disable();
	auto sampler1 = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler1->varSamplingDistance()->setValue(0.005);
	sampler1->setVisible(false);
	cube1->connect(sampler1->importShape());
	auto initialParticles1 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler1->statePointSet()->promoteOuput()->connect(initialParticles1->inPoints());

	auto cube2 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube2->varLocation()->setValue(Vec3f(-0.125, 0.125, 0.125));
	cube2->varLength()->setValue(Vec3f(0.15, 0.15, 0.15));
	cube2->graphicsPipeline()->disable();
	auto sampler2 = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler2->varSamplingDistance()->setValue(0.005);
	sampler2->setVisible(false);
	cube2->connect(sampler2->importShape());
	auto initialParticles2 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler2->statePointSet()->promoteOuput()->connect(initialParticles2->inPoints());

	auto cube3 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube3->varLocation()->setValue(Vec3f(0.125, 0.125, -0.125));
	cube3->varLength()->setValue(Vec3f(0.15, 0.15, 0.15));
	cube3->graphicsPipeline()->disable();
	auto sampler3 = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler3->varSamplingDistance()->setValue(0.005);
	sampler3->setVisible(false);
	cube3->connect(sampler3->importShape());
	auto initialParticles3 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler3->statePointSet()->promoteOuput()->connect(initialParticles3->inPoints());

	auto cube4 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube4->varLocation()->setValue(Vec3f(-0.125, 0.125, -0.125));
	cube4->varLength()->setValue(Vec3f(0.15, 0.15, 0.15));
	cube4->graphicsPipeline()->disable();
	auto sampler4 = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler4->varSamplingDistance()->setValue(0.005);
	sampler4->setVisible(false);
	cube4->connect(sampler4->importShape());
	auto initialParticles4 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler4->statePointSet()->promoteOuput()->connect(initialParticles4->inPoints());



	auto fluid = scn->addNode(std::make_shared<DualParticleFluid<DataType3f>>());
	fluid->varReshuffleParticles()->setValue(true);
	initialParticles1->connect(fluid->importInitialStates());
	initialParticles2->connect(fluid->importInitialStates());
	initialParticles3->connect(fluid->importInitialStates());
	initialParticles4->connect(fluid->importInitialStates());

	//Create a boundary
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.0f, 1.0f, 0.0f));
	cubeBoundary->varLength()->setValue(Vec3f(0.5f, 2.0f, 0.5f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(container->importVolumes());

	fluid->connect(container->importParticleSystems());

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->varPointSize()->setValue(0.0035f);
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	fluid->statePointSet()->connect(ptRender->inPointSet());
	colorMapper->outColor()->connect(ptRender->inColor());
	fluid->graphicsPipeline()->pushModule(ptRender);

	// A simple color bar widget for node
	auto colorBar = std::make_shared<ImColorbar>();
	colorBar->varMax()->setValue(5.0f);
	colorBar->varFieldName()->setValue("Velocity");
	calculateNorm->outNorm()->connect(colorBar->inScalar());
	// add the widget to app
	fluid->graphicsPipeline()->pushModule(colorBar);

	auto vpRender = std::make_shared<GLPointVisualModule>();
	vpRender->setColor(Color(1, 1, 0));
	vpRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	fluid->stateVirtualPointSet()->connect(vpRender->inPointSet());
	vpRender->varPointSize()->setValue(0.0005);
	fluid->graphicsPipeline()->pushModule(vpRender);

	return scn;
}

int main()
{
	GlfwApp window;
	window.setSceneGraph(createScene());
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}


