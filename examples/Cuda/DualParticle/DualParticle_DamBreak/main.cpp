#include <UbiApp.h>
#include <SceneGraph.h>

#include <Volume/BasicShapeToVolume.h>
#include <Multiphysics/VolumeBoundary.h>

#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>


#include <ParticleSystem/MakeParticleSystem.h>
#include <BasicShapes/CubeModel.h>
#include <Samplers/ShapeSampler.h>
#include <ParticleSystem/Emitters/SquareEmitter.h>
#include <ParticleSystem/ParticleFluid.h>

using namespace std;
using namespace dyno;

bool useVTK = false;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(3.0, 3.0, 3.0));
	scn->setLowerBound(Vec3f(-3.0, -3.0, -3.0));

	auto cube1 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube1->varLocation()->setValue(Vec3f(0.47, 0.1, 0.13));
	cube1->varLength()->setValue(Vec3f(0.2, 0.2, 0.2));
	cube1->graphicsPipeline()->disable();
	auto sampler1 = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler1->varSamplingDistance()->setValue(0.005);
	sampler1->setVisible(false);
	cube1->connect(sampler1->importShape());
	auto initialParticles1 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler1->statePointSet()->promoteOuput()->connect(initialParticles1->inPoints());

	auto cube4 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube4->varLocation()->setValue(Vec3f(0.13, 0.1, 0.47));
	cube4->varLength()->setValue(Vec3f(0.2, 0.2, 0.2));
	cube4->graphicsPipeline()->disable();
	auto sampler4 = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler4->varSamplingDistance()->setValue(0.005);
	sampler4->setVisible(false);
	cube4->connect(sampler4->importShape());
	auto initialParticles4 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler4->statePointSet()->promoteOuput()->connect(initialParticles4->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->varIncompressibilitySolver()->getDataPtr()->setCurrentKey(ParticleFluid<DataType3f>::FissionDP);
	fluid->setDt(0.001);
	fluid->varSmoothingLength()->setValue(2.4);
	fluid->varReshuffleParticles()->setValue(true);
	initialParticles1->connect(fluid->importInitialStates());
	initialParticles4->connect(fluid->importInitialStates());

	fluid->graphicsPipeline()->clear();

	//Create a boundary
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.3f, 0.3f, 0.3f));
	cubeBoundary->varLength()->setValue(Vec3f(0.6f, 0.6f, 0.6f));
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
	ptRender->varBaseColor()->setValue(Color(1, 0, 0));
	ptRender->varPointSize()->setValue(0.0028f);
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	fluid->statePointSet()->connect(ptRender->inPointSet());
	colorMapper->outColor()->connect(ptRender->inColor());
	fluid->graphicsPipeline()->pushModule(ptRender);

	// A simple color bar widget for node
	auto colorBar = std::make_shared<ImColorbar>();
	colorBar->varMax()->setValue(2.0f);
	colorBar->varFieldName()->setValue("Velocity");
	calculateNorm->outNorm()->connect(colorBar->inScalar());
	// add the widget to app
	fluid->graphicsPipeline()->pushModule(colorBar);

	return scn;
}

int main()
{
//	QtApp window;
//	GlfwApp window;

	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createScene());
	app.initialize(2048, 1080);


	auto cam = app.renderWindow()->getCamera();
	cam->setEyePos(Vec3f(0.319212, 0.933322, 1.04517));
	cam->setTargetPos(Vec3f(0.307528, -0.0511578, 0.0174742));

	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->showGround = false;
		renderer->setUseEnvmapBackground(false);
	}

	app.mainLoop();

	return 0;
}


