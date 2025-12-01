#include <GlfwApp.h>
#include <SceneGraph.h>

#include "BasicShapes/CircleModel2D.h"
#include "Volume/AdaptiveVolumeFromBasicShape2D.h"
#include "Volume/GLAdaptiveGridVisualNode2D.h"
#include "EulerFluid/GLEuleSimVisualNode2D.h"
#include "EulerFluid/EulerianSimSplitting2D.h"
#include "StaticBoundaryCircle.h"

//#include "Volume/SparseMarchingCubes.h"
#include "GLSurfaceVisualModule.h"
#include "Volume/GLAdaptiveXYPlaneVisualNode.h"
#include "GLWireframeVisualModule.h"

#include "BasicShapes/CubeModel.h"
#include "Samplers/ShapeSampler.h"
#include "ParticleSystem/MakeParticleSystem.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/IterativeFrictionSolver.h"

#include "Multiphysics/VolumeBoundary.h"

#include "Collision/NeighborPointQuery.h"
#include "Auxiliary/DataSource.h"
#include "Module/CalculateNorm.h"
#include "GLPointVisualModule.h"
#include "ColorMapping.h"
#include "ImColorbar.h"
//#include "ABCExporter\ParticleWriterABC.h"
//#include "TriangleMeshWriter.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-1.0f, 0.0f, -0.1f));
	scn->setUpperBound(Vec3f(1.0f, 2.0f, 0.1f));

	//create a cube.
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0, 0.825, 0.0));
	cube->varLength()->setValue(Vec3f(0.40, 1.6, 0.004));
	cube->setVisible(false);

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.002);
	sampler->setVisible(false);
	cube->connect(sampler->importShape());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	initialParticles->connect(fluid->importInitialStates());

	fluid->animationPipeline()->clear();

	auto smoothingLength = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
	smoothingLength->varValue()->setValue(Real(0.004));

	auto samplingDistance = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
	samplingDistance->varValue()->setValue(Real(0.002));

	auto integrator = std::make_shared<ParticleIntegrator<DataType3f>>();
	fluid->stateTimeStep()->connect(integrator->inTimeStep());
	fluid->statePosition()->connect(integrator->inPosition());
	fluid->stateVelocity()->connect(integrator->inVelocity());
	fluid->animationPipeline()->pushModule(integrator);

	auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
	smoothingLength->outFloating()->connect(nbrQuery->inRadius());
	fluid->statePosition()->connect(nbrQuery->inPosition());
	fluid->animationPipeline()->pushModule(nbrQuery);

	auto density = std::make_shared<IterativeFrictionSolver<DataType3f>>();
	density->varIterationNumber()->setValue(20);
	smoothingLength->outFloating()->connect(density->inSmoothingLength());
	samplingDistance->outFloating()->connect(density->inSamplingDistance());
	fluid->stateTimeStep()->connect(density->inTimeStep());
	fluid->statePosition()->connect(density->inPosition());
	fluid->stateVelocity()->connect(density->inVelocity());
	nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
	fluid->animationPipeline()->pushModule(density);

	auto viscosity = std::make_shared<ImplicitViscosity<DataType3f>>();
	viscosity->varViscosity()->setValue(Real(0.5));
	fluid->stateTimeStep()->connect(viscosity->inTimeStep());
	smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
	fluid->statePosition()->connect(viscosity->inPosition());
	fluid->stateVelocity()->connect(viscosity->inVelocity());
	nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
	fluid->animationPipeline()->pushModule(viscosity);

	//Create a boundary
	auto boundary = std::make_shared<StaticBoundaryCircle<DataType3f>>();
	boundary->varNormalFriction()->setValue(1.0f);
	boundary->varTangentialFriction()->setValue(1.0f);
	boundary->varPlaneTangentialFriction()->setValue(0.1f);
	boundary->varCenter()->setValue(Vec3f(0.0, 0.9, 0.0));
	boundary->varRadius()->setValue(0.9f);
	boundary->varDx()->setValue(0.004f);
	fluid->statePosition()->connect(boundary->inPosition());
	fluid->stateVelocity()->connect(boundary->inVelocity());
	fluid->animationPipeline()->pushModule(boundary);

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	//ptRender->setColor(Color(0.76, 0.7, 0.5));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	fluid->statePointSet()->connect(ptRender->inPointSet());
	colorMapper->outColor()->connect(ptRender->inColor());
	fluid->graphicsPipeline()->pushModule(ptRender);

	// A simple color bar widget for node
	auto colorBar = std::make_shared<ImColorbar>();
	colorBar->varMax()->setValue(5.0f); 
	colorBar->varFieldName()->setValue("Velocity");
	calculateNorm->outNorm()->connect(colorBar->inScalar());
	fluid->graphicsPipeline()->pushModule(colorBar);

	//auto abcWriter = std::make_shared<ParticleWriterABC<DataType3f>>();
	//abcWriter->setNamePrefix("particles");
	//abcWriter->setOutputPath("D:/ACFD/Quicksand/abc");
	//abcWriter->varFrameStep()->setValue(5);
	//abcWriter->varEnd()->setValue(10000);
	//fluid->stateFrameNumber()->connect(abcWriter->inFrameNumber());
	//fluid->statePosition()->connect(abcWriter->inPosition());
	//fluid->stateVelocity()->connect(abcWriter->inVelocity());
	////calculateNorm->outNorm()->connect(abcWriter->inColor());
	//fluid->animationPipeline()->pushModule(abcWriter);


	auto circle = scn->addNode(std::make_shared<CircleModel2D<DataType3f>>());
	circle->varRadius()->setValue(0.9f);
	circle->varCenter2D()->setValue(Vec2f(0.0, 0.9));

	auto AGrid = scn->addNode(std::make_shared<AdaptiveVolumeFromBasicShape2D<DataType3f>>());
	circle->connect(AGrid->importShapes());
	AGrid->varNarrowWidth()->setValue(0.01f);
	AGrid->varDx()->setValue(0.004f);
	AGrid->varLevelNum()->setValue(9);
	fluid->statePosition()->connect(AGrid->inParticles());

	auto esim = scn->addNode(std::make_shared<EulerianSimSplitting2D<DataType3f>>());
	AGrid->stateAGridSet()->promoteOuput()->connect(esim->inAdaptiveVolume2D());
	esim->inRadius()->setValue(0.9f);
	esim->inCenter()->setValue(Vec2f(0.0, 0.9));
	fluid->stateVelocity()->connect(esim->statePVelocity());
	fluid->statePosition()->connect(esim->statePPosition());
	samplingDistance->outFloating()->connect(esim->varSamplingDistance());
	esim->varSandDensity()->setValue(5000.0f);
	esim->varUpdateCoefficient()->setValue(0.1f);
	 
	auto qtVisualizer = scn->addNode(std::make_shared<GLAdaptiveGridVisualNode2D<DataType3f>>());
	AGrid->stateAGridSet()->connect(qtVisualizer->inAdaptiveVolume());
	qtVisualizer->varEType()->setCurrentKey(GLAdaptiveGridVisualNode2D<DataType3f>::Quadtree_Edge);
	qtVisualizer->varPPlane()->setCurrentKey(GLAdaptiveGridVisualNode2D<DataType3f>::XY_Plane);

	//auto triWriter = std::make_shared<TriangleMeshWriter<DataType3f>>();
	//triWriter->setOutputPath("D:/ACFD/Quicksand/obj/");
	//triWriter->setNamePrefix("grid");
	//triWriter->varFrameStep()->setValue(5);
	//triWriter->varEnd()->setValue(10000);
	//pRender->stateGrids()->promoteOuput()->connect(triWriter->inTopology());
	//triWriter->varOutputType()->setCurrentKey(TriangleMeshWriter<DataType3f>::Edges);
	//fluid->stateFrameNumber()->connect(triWriter->inFrameNumber());
	//fluid->animationPipeline()->pushModule(triWriter);


	return scn;
}

int main()
{
	GlfwApp window;

	window.setSceneGraph(createScene());
	window.initialize(1280, 768);

	window.mainLoop();

	return 0;
}


