#include <QtApp.h>
#include <SceneGraph.h>
#include <Log.h>
#include <ParticleSystem/ParticleFluid.h>
#include <RigidBody/RigidBody.h>
#include "ParticleSystem/MakeParticleSystem.h"
#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>
#include <BasicShapes/CubeModel.h>
#include <Samplers/ShapeSampler.h>
#include <GLSurfaceVisualModule.h>
#include "Node.h"
#include "Topology/TriangleSet.h"

#include <Multiphysics/VolumeBoundary.h>

//ParticleSystem
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/SemiImplicitDensitySolver.h"
#include "ParticleSystem/Module/IterativeDensitySolver.h"
#include "ParticleSystem/Module/ApproximateImplicitViscosity.h"
#include <ParticleSystem/Emitters/SquareEmitter.h>
#include "Waterfall.h"
//Framework
#include "Auxiliary/DataSource.h"

//Topology
#include "Collision/NeighborPointQuery.h"
#include "Collision/NeighborTriangleQuery.h"
#include "Topology/TriangleSet.h"
#include "Mapping/TextureMeshToTriangleSet.h"
#include "SemiAnalyticalScheme/TriangularMeshBoundary.h"
#include "SemiAnalyticalScheme/TriangularMeshConstraint.h"

//ParticleOutput
#include "ABCExporter/ParticleWriterABC.h"
#include "ObjIO/PLYexporter.h"

//OBJ
#include "ObjIO/ObjLoader.h"


//
#include "GltfLoader.h"

using namespace std;
using namespace dyno;

#define PARTICLE_SPACING 0.005f 
#define SIISPH_MODE
#define KAPPA 80.0f
#define SMOOTHINGLEHGTH PARTICLE_SPACING*2.5f

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f, -10.0f, 0.0f));
	scn->setUpperBound(Vec3f(6.629, 3.805f, 2.224));
	scn->setLowerBound(Vec3f(-4.793, 0.145f, -2.224));


	auto emitter = scn->addNode(std::make_shared<Waterfall<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(-2.37f, 2.99f, -0.07f));
	emitter->varRotation()->setValue(Vec3f(-0.0f, -5.0f, 50.0f));
	emitter->varSamplingDistance()->setValue(PARTICLE_SPACING);
	emitter->varVelocityMagnitude()->setValue(2.5);
	emitter->varHeight()->setValue(0.20f);
	emitter->varWidth()->setValue(0.1f);
	////////Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(-1.9, 2.78, -0.04));
	cube->varRotation()->setValue(Vec3f(0.0f, 0.0f, 0.0f));
	cube->varLength()->setValue(Vec3f(0.02, 0.02, 0.02));
	
	cube->graphicsPipeline()->disable();

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(PARTICLE_SPACING);
	sampler->graphicsPipeline()->disable();

	cube->connect(sampler->importShape());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());


	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	emitter->connect(fluid->importParticleEmitters());

	//seaSide->graphicsPipeline()->clear();

	auto gltf = scn->addNode(std::make_shared <GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "scene/gltf/Waterfall/Waterfall.gltf");


	auto gltfGrass = scn->addNode(std::make_shared <GltfLoader<DataType3f>>());
	gltfGrass->varFileName()->setValue(getAssetPath() + "scene/gltf/Waterfall/grass.gltf");


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

#ifdef SIISPH_MODE 
		auto density = std::make_shared<SemiImplicitDensitySolver<DataType3f>>();
		density->varKappa()->setValue(KAPPA);
		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());
		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);
#else

		auto density = std::make_shared<IterativeDensitySolver<DataType3f>>();
		density->varKappa()->setValue(KAPPA);
		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());
		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);

#endif
		auto viscosity = std::make_shared<ImplicitViscosity<DataType3f>>();
		viscosity->varViscosity()->setValue(Real(5.0));
		fluid->stateTimeStep()->connect(viscosity->inTimeStep());
		smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
		samplingDistance->outFloating()->connect(viscosity->inSamplingDistance());
		fluid->statePosition()->connect(viscosity->inPosition());
		fluid->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		fluid->animationPipeline()->pushModule(viscosity);

		//auto viscosity = std::make_shared<ApproximateImplicitViscosity<DataType3f>>();
		//viscosity->setViscosityValue(Real(5.0));
		//fluid->stateTimeStep()->connect(viscosity->inTimeStep());
		//smoothingLength->outFloating()->connect(viscosity->varSmoothingLength());
		//samplingDistance->outFloating()->connect(viscosity->varSamplingDistance());
		//fluid->statePosition()->connect(viscosity->inPosition());
		//fluid->stateVelocity()->connect(viscosity->inVelocity());
		//nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		//fluid->animationPipeline()->pushModule(viscosity);
	}

	initialParticles->connect(fluid->importInitialStates());

	auto texMeshConverter = scn->addNode(std::make_shared<TextureMeshToTriangleSetNode<DataType3f>>());
	gltf->stateTextureMesh()->connect(texMeshConverter->inTextureMesh());

	auto pm_collide = scn->addNode(std::make_shared <TriangularMeshBoundary<DataType3f>>());
	fluid->connect(pm_collide->importParticleSystems());
	texMeshConverter->outTriangleSet()->connect(pm_collide->inTriangleSet());



	//Create a boundary
// 	auto boundary = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
// 	boundary->loadCube(Vec3f(-2.5, 0.3, -2.24), Vec3f(3.5, 5.0, 2.11), 0.02, true);
// 	boundary->loadCube(Vec3f(-2.7, -0.3, -2.4), Vec3f(3.7, 6.0, 2.3), 0.02, true);
// 	boundary->varCubeVertex_lo()->setValue(Vec3f(-2.5, 0.3, -1.9));
// 	boundary->varCubeVertex_hi()->setValue(Vec3f(3.5, 5.0, 1.9));
// 	boundary->loadCube(Vec3f(-2.4, 1.6, -1.4), Vec3f(0.16, 3.6, -1.24), 0.02, false);
// 	boundary->loadCube(Vec3f(-2.4, 1.6, 0.85), Vec3f(0.16, 3.6, 0.9), 0.02, false);
// 	boundary->loadCube(Vec3f(-2.6, 0.0, -2.25), Vec3f(0.15, 1.92, 2.22), 0.02, false);
//	fluid->connect(boundary->importParticleSystems());

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

// 	auto output = std::make_shared<ParticleWriterABC<DataType3f>>();
// 	fluid->stateFrameNumber()->connect(output->inFrameNumber());
// 	output->varInterval()->setValue(20);
// 	fluid->statePointSet()->connect(output->inPointSet());
// 	calculateNorm->outNorm()->connect(output->inColor());
// 	output->varOutputPath()->setValue(FilePath("D:/DATA/Cache/"));
// 	//output->varPrefix()->setValue("fluid_");
// 	fluid->animationPipeline()->pushModule(output);


	//auto plyoutput = scn->addNode(std::make_shared <PlyExporter<DataType3f>>());
	//fluid->statePointSet()->connect(plyoutput->inTopology());


	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	ptRender->varPointSize()->setValue(0.2 * PARTICLE_SPACING);
	fluid->statePointSet()->connect(ptRender->inPointSet());
	colorMapper->outColor()->connect(ptRender->inColor());

	fluid->graphicsPipeline()->pushModule(ptRender);

	//// A simple color bar widget for node
	//auto colorBar = std::make_shared<ImColorbar>();
	//colorBar->varMax()->setValue(5.0f);
	//colorBar->varFieldName()->setValue("Velocity");
	//calculateNorm->outNorm()->connect(colorBar->inScalar());
	//// add the widget to app
	//fluid->graphicsPipeline()->pushModule(colorBar);
	return scn;
}

int main()
{
	QtApp app;

	app.setSceneGraph(createScene());
	// window.createWindow(2048, 1152);
	app.initialize(1024, 768);

	// setup envmap
	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setUseEnvmapBackground(false);

		renderer->bgColor0 = { 1, 1, 1 };
		renderer->bgColor1 = { 1, 1, 1 };

		renderer->planeColor = { 1,1,1,1 };
		renderer->rulerColor = { 1,1,1,1 };
		renderer->showGround = false;

		renderer->setEnvmapScale(0.2f);

		auto& light = app.renderWindow()->getRenderParams().light;
		light.mainLightScale = 10;

	}


	app.mainLoop();

	return 0;
}