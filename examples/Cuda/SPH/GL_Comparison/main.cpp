#include <UbiApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <ParticleSystem/Module/SemiImplicitDensitySolver.h>
#include <ParticleSystem/Module/ImplicitISPH.h>
#include <ParticleSystem/Module/IterativeDensitySolver.h>
#include <ParticleSystem/Module/DivergenceFreeSphSolver.h>
#include <ParticleSystem/Module/ImplicitViscosity.h>
#include <ParticleSystem/Module/ParticleIntegrator.h>

#include <Collision/NeighborPointQuery.h>

#include <Volume/VolumeLoader.h>
#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <ColorMapping.h>
#include <ImColorbar.h>

#include <BasicShapes/SphereModel.h>
#include <BasicShapes/CubeModel.h>

#include <StaticMeshLoader.h>

#include <Samplers/ShapeSampler.h>

#include <Auxiliary/DataSource.h>

using namespace std;
using namespace dyno;

#define PARTICLE_SPACING 0.005f 
#define SMOOTHINGLEHGTH PARTICLE_SPACING*2.0

//#define SISPH
#define IISPH
//#define DFSPH
//#define PBF

#define TIME_STEP_SIZE 0.001

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f));
	scn->setUpperBound(Vec3f(2, 2, 2));
	scn->setLowerBound(Vec3f(-2, -2, -2));

	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0));
	cube->varScale()->setValue(Vec3f(0.02f, 0.5f, 0.5f));

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(PARTICLE_SPACING);
	sampler->setVisible(false);

	cube->connect(sampler->importShape());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	initialParticles->varInitialVelocity()->setValue(Vec3f(0.0f));

	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	//Create the second sphere
	auto sphere2 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	sphere2->varLocation()->setValue(Vec3f(-0.3, -0.0, 0.0));
	sphere2->varRadius()->setValue(0.05f);

	//Create a sampler
	auto sampler2 = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler2->varSamplingDistance()->setValue(PARTICLE_SPACING);
	sampler2->setVisible(false);

	sphere2->connect(sampler2->importShape());

	auto initialParticles2 = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	initialParticles2->varInitialVelocity()->setValue(Vec3f(1.0f, 0.0f, 0.0f));

	sampler2->statePointSet()->promoteOuput()->connect(initialParticles2->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->setDt(TIME_STEP_SIZE);
	fluid->varReshuffleParticles()->setValue(true);
	initialParticles->connect(fluid->importInitialStates());
	initialParticles2->connect(fluid->importInitialStates());

	{
		fluid->animationPipeline()->clear();
		auto smoothingLength = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
		smoothingLength->varValue()->setValue(Real(SMOOTHINGLEHGTH));

		auto samplingDistance = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
		samplingDistance->varValue()->setValue(Real(PARTICLE_SPACING));

		auto integrator = std::make_shared<ParticleIntegrator<DataType3f>>();
		fluid->stateTimeStep()->connect(integrator->inTimeStep());
		fluid->statePosition()->connect(integrator->inPosition());
		fluid->stateVelocity()->connect(integrator->inVelocity());
		fluid->animationPipeline()->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		//nbrQuery->varSizeLimit()->setValue(30);
		smoothingLength->outFloating()->connect(nbrQuery->inRadius());
		fluid->statePosition()->connect(nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(nbrQuery);

		//auto nbrQueryTri = std::make_shared<NeighborTriangleQuery<DataType3f>>();
		//smoothingLength->outFloating()->connect(nbrQueryTri->inRadius());
		//fluid->statePosition()->connect(nbrQueryTri->inPosition());
		//fluid->stateTriangleVertex()->connect(nbrQueryTri->inTriPosition());
		//fluid->stateTriangleIndex()->connect(nbrQueryTri->inTriangles());
		//fluid->animationPipeline()->pushModule(nbrQueryTri);

#ifdef SISPH
		auto density = std::make_shared<SemiImplicitDensitySolver<DataType3f>>();
		density->varIterationNumber()->setValue(10);
		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());
		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);
#endif

#ifdef PBF
		auto density = std::make_shared<IterativeDensitySolver<DataType3f>>();
		density->varIterationNumber()->setValue(10);
		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());
		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);

#endif

#ifdef DFSPH
		auto density = std::make_shared<DivergenceFreeSphSolver<DataType3f>>();
		density->varMaxIterationNumber()->setValue(10);
		density->varDivergenceSolverDisabled()->setValue(true);
		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());
		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);
#endif

#ifdef IISPH
		auto density = std::make_shared<ImplicitISPH<DataType3f>>();
		density->varIterationNumber()->setValue(10);
		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());
		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);
#endif

		auto viscosity = std::make_shared<ImplicitViscosity<DataType3f>>();
		viscosity->varViscosity()->setValue(Real(0.1));
		fluid->stateTimeStep()->connect(viscosity->inTimeStep());
		smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
		samplingDistance->outFloating()->connect(viscosity->inSamplingDistance());
		fluid->statePosition()->connect(viscosity->inPosition());
		fluid->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		fluid->animationPipeline()->pushModule(viscosity);
	}

	//Replace the default rendering modules
	{
		fluid->graphicsPipeline()->clear();

		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		fluid->stateVelocity()->connect(calculateNorm->inVec());
		fluid->graphicsPipeline()->pushModule(calculateNorm);

		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(1.0f);
		calculateNorm->outNorm()->connect(colorMapper->inScalar());
		fluid->graphicsPipeline()->pushModule(colorMapper);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(1, 0, 0));
		ptRender->varPointSize()->setValue(0.8 * PARTICLE_SPACING);
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		fluid->statePointSet()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		fluid->graphicsPipeline()->pushModule(ptRender);

		// A simple color bar widget for node
		auto colorBar = std::make_shared<ImColorbar>();
		colorBar->varMax()->setValue(1.0f);
		colorBar->varFieldName()->setValue("Velocity");
		calculateNorm->outNorm()->connect(colorBar->inScalar());
		// add the widget to app
		fluid->graphicsPipeline()->pushModule(colorBar);
	}

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_GLFW);

	app.setSceneGraph(createScene());
	// window.createWindow(2048, 1152);
	app.initialize(1024, 1024);
	auto cam = app.renderWindow()->getCamera();
	cam->setEyePos(Vec3f(0.27f, 0.35f, 0.99f));

	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->showGround = false;
		renderer->setUseEnvmapBackground(false);
	}

	app.mainLoop();

	return 0;
}


