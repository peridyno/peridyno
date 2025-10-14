#include <GlfwApp.h>
#include <SceneGraph.h>

///Particle Emitter
#include <ParticleSystem/Emitters/PoissonEmitter.h>

///Fluid Solver
#include <DualParticleSystem/DualParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>

///Renderer
#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(3.0, 3, 3.0));
	scn->setLowerBound(Vec3f(-3.0, 0, -3.0));

	auto emitter = scn->addNode(std::make_shared<PoissonEmitter<DataType3f>>());
	emitter->varSpacing()->setValue(0);
	emitter->varRotation()->setValue(Vec3f(0.0f, 0.0f, -90.0f));
	emitter->varSamplingDistance()->setValue(0.008f);
	emitter->varEmitterShape()->getDataPtr()->setCurrentKey(1);
	emitter->varWidth()->setValue(0.1f);
	emitter->varHeight()->setValue(0.1f);
	emitter->varVelocityMagnitude()->setValue(1.5);
	emitter->varLocation()->setValue(Vec3f(0.2f, 0.5f, 0.0f));

	auto emitter2 = scn->addNode(std::make_shared<PoissonEmitter<DataType3f>>());
	emitter2->varSpacing()->setValue(0);
	emitter2->varRotation()->setValue(Vec3f(0.0f, 0.0f, 90.0f));
	emitter2->varSamplingDistance()->setValue(0.008f);
	emitter2->varEmitterShape()->getDataPtr()->setCurrentKey(1);
	emitter2->varWidth()->setValue(0.1f);
	emitter2->varHeight()->setValue(0.1f);
	emitter2->varVelocityMagnitude()->setValue(1.5);
	emitter2->varLocation()->setValue(Vec3f(-0.2f, 0.5f, -0.0f));

	auto fluid = scn->addNode(std::make_shared<DualParticleFluid<DataType3f>>(
		DualParticleFluid<DataType3f>::FissionFusionStrategy));
	emitter->connect(fluid->importParticleEmitters());
	emitter2->connect(fluid->importParticleEmitters());

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
	vpRender->varPointSize()->setValue(0.001);
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


