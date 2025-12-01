#include <UbiApp.h>
#include <SceneGraph.h>

#include "StaticBoundaryEmitter.h"

#include "ConstrainParticleEmitterZAxis.h"

#include "BasicShapes/CircleModel2D.h"
#include "Volume/AdaptiveVolumeFromBasicShape2D.h"
#include "Volume/GLAdaptiveGridVisualNode2D.h"
#include "EulerFluid/GLEuleSimVisualNode2D.h"
#include "EulerFluid/EulerianSimSplitting2D.h"
#include "StaticBoundaryCircle.h"

#include "ParticleSystem/Emitters/SquareEmitter.h"
#include "ParticleSystem/Emitters/PoissonEmitter.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/IterativeFrictionSolver.h"

#include "Collision/NeighborPointQuery.h"
#include "Auxiliary/DataSource.h"
#include "Module/CalculateNorm.h"
#include "GLPointVisualModule.h"
#include "ColorMapping.h"
#include "ImColorbar.h"
#include "ABCExporter/ParticleWriterABC.h"
#include "TriangleMeshWriter.h"
#include "ParticleWriter.h"


using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f, -1.0f, 0.0f));
	scn->setLowerBound(Vec3f(-1.0f, 0.0f, -0.1f));
	scn->setUpperBound(Vec3f(1.0f, 2.0f, 0.1f));

	////Create a particle emitter
	//auto emitter = scn->addNode(std::make_shared<SquareEmitter<DataType3f>>());
	//emitter->varLocation()->setValue(Vec3f(0.0, 1.8f, 0.0));
	//emitter->varVelocityMagnitude()->setValue(1.0f);
	//emitter->varSamplingDistance()->setValue(0.01f);
	//emitter->varWidth()->setValue(0.1f);
	//emitter->varHeight()->setValue(0.1f);

	auto emitter = scn->addNode(std::make_shared<PoissonEmitter<DataType3f>>());
	emitter->varSamplingDistance()->setValue(0.028f);
	emitter->varVelocityMagnitude()->setValue(1.0f);
	emitter->varLocation()->setValue(Vec3f(0.0f, 1.8f, 0.0f));
	emitter->varWidth()->setValue(0.1f);
	emitter->varHeight()->setValue(0.1f);
	emitter->varEmitterShape()->setCurrentKey(PoissonEmitter<DataType3f>::Square);

	auto constemi = scn->addNode(std::make_shared<ConstrainParticleEmitterZAxis<DataType3f>>());
	constemi->varZdx()->setValue(0.004);
	emitter->connect(constemi->importParticleEmitters());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	emitter->connect(fluid->importParticleEmitters());

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
	viscosity->varViscosity()->setValue(Real(0.2));
	fluid->stateTimeStep()->connect(viscosity->inTimeStep());
	smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
	fluid->statePosition()->connect(viscosity->inPosition());
	fluid->stateVelocity()->connect(viscosity->inVelocity());
	nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
	fluid->animationPipeline()->pushModule(viscosity);

	//Create a boundary
	auto boundary = std::make_shared<StaticBoundaryCircle<DataType3f>>();
	boundary->varNormalFriction()->setValue(1.0f);
	boundary->varTangentialFriction()->setValue(0.2f);
	boundary->varPlaneTangentialFriction()->setValue(0.1f);
	boundary->varCenter()->setValue(Vec3f(0.0, 0.9, 0.0));
	boundary->varRadius()->setValue(0.9f);
	boundary->varDx()->setValue(0.004f);
	fluid->statePosition()->connect(boundary->inPosition());
	fluid->stateVelocity()->connect(boundary->inVelocity());
	fluid->animationPipeline()->pushModule(boundary);

	auto eboundary = std::make_shared<StaticBoundaryEmitter<DataType3f>>();
	emitter->varLocation()->connect(eboundary->varLocation());
	emitter->varWidth()->connect(eboundary->varXWidth());
	eboundary->varYHigh()->setValue(0.04f);
	fluid->stateVelocity()->connect(eboundary->inVelocity());
	fluid->statePosition()->connect(eboundary->inPosition());
	fluid->animationPipeline()->pushModule(eboundary);

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
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
	//abcWriter->setOutputPath("D:/ACFD/QuicksandEmitter/abc");
	//abcWriter->varFrameStep()->setValue(20);
	//abcWriter->varEnd()->setValue(100000);
	//fluid->stateFrameNumber()->connect(abcWriter->inFrameNumber());
	//fluid->statePosition()->connect(abcWriter->inPosition());
	//fluid->stateVelocity()->connect(abcWriter->inVelocity());
	////calculateNorm->outNorm()->connect(abcWriter->inColor());
	//fluid->animationPipeline()->pushModule(abcWriter);

	//auto ptcWriter = std::make_shared<ParticleWriter<DataType3f>>();
	//ptcWriter->setNamePrefix("objPtc");
	//ptcWriter->setOutputPath("D:/ACFD/QuicksandEmitter/txt");
	//ptcWriter->varInterval()->setValue(2000);
	//fluid->statePointSet()->connect(ptcWriter->inPointSet());
	//fluid->animationPipeline()->pushModule(ptcWriter);


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
	AGrid->stateAGridSet()->connect(esim->inAdaptiveVolume2D());
	esim->inRadius()->setValue(0.9f);
	esim->inCenter()->setValue(Vec2f(0.0, 0.9));
	fluid->stateVelocity()->connect(esim->statePVelocity());
	fluid->statePosition()->connect(esim->statePPosition());
	samplingDistance->outFloating()->connect(esim->varSamplingDistance());
	esim->varSandDensity()->setValue(5000.0f);
	esim->varUpdateCoefficient()->setValue(0.03f);
	//esim->varSandDensity()->setValue(5000.0f);
	//esim->varUpdateCoefficient()->setValue(0.1f);

	auto qtVisualizer = scn->addNode(std::make_shared<GLAdaptiveGridVisualNode2D<DataType3f>>());
	AGrid->stateAGridSet()->connect(qtVisualizer->inAdaptiveVolume());
	qtVisualizer->varEType()->setCurrentKey(GLAdaptiveGridVisualNode2D<DataType3f>::Quadtree_Edge);
	qtVisualizer->varPPlane()->setCurrentKey(GLAdaptiveGridVisualNode2D<DataType3f>::XY_Plane);

	//auto triWriter = std::make_shared<TriangleMeshWriter<DataType3f>>();
	//triWriter->setOutputPath("D:/ACFD/QuicksandEmitter/obj/");
	//triWriter->setNamePrefix("grid");
	//triWriter->varFrameStep()->setValue(20);
	//triWriter->varEnd()->setValue(100000);
	//pRender->stateGrids()->promoteOuput()->connect(triWriter->inTopology());
	//triWriter->varOutputType()->setCurrentKey(TriangleMeshWriter<DataType3f>::Edges);
	//fluid->stateFrameNumber()->connect(triWriter->inFrameNumber());
	//fluid->animationPipeline()->pushModule(triWriter);

	return scn;
}

int main()
{
	UbiApp window(GUIType::GUI_QT);

	window.setSceneGraph(createScene());
	window.initialize(1280, 768);

	window.mainLoop();

	return 0;
}


