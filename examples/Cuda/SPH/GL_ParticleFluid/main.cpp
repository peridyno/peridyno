#include <UbiApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>

#include <Volume/VolumeLoader.h>
#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <ColorMapping.h>
#include <ImColorbar.h>

#include <BasicShapes/CubeModel.h>

#include <StaticMeshLoader.h>

#include <Samplers/ShapeSampler.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(1.5, 1, 1.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));

	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.6, 0.6, 0.5));
	cube->varLength()->setValue(Vec3f(0.5, 0.5, 0.5));
	cube->graphicsPipeline()->disable();

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->graphicsPipeline()->disable();

	cube->connect(sampler->importShape());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());

	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->varReshuffleParticles()->setValue(true);
	initialParticles->connect(fluid->importInitialStates());

	auto volLoader = scn->addNode(std::make_shared<VolumeLoader<DataType3f>>());
	volLoader->varFileName()->setValue(getAssetPath() + "bowl/bowl.sdf");

	auto volBoundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	volLoader->connect(volBoundary->importVolumes());

	//Connect ParticleFluid to VolumeBoundary
	fluid->connect(volBoundary->importParticleSystems());

	//Create a cube for boundary
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.5f, 1.0f, 0.5f));
	cubeBoundary->varLength()->setValue(Vec3f(2.0f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());
	cube2vol->connect(volBoundary->importVolumes());

	auto staticMesh = scn->addNode(std::make_shared<StaticMeshLoader<DataType3f>>());
	staticMesh->varFileName()->setValue(getAssetPath() + "bowl/bowl.obj");

	fluid->graphicsPipeline()->clear();

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
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

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_QT);

	app.setSceneGraph(createScene());
	// window.createWindow(2048, 1152);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


