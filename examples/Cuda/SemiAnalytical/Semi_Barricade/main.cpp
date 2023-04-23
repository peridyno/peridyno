#include "GlfwGUI/GlfwApp.h"

#include "SceneGraph.h"
#include "Log.h"

#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"

#include "Topology/TriangleSet.h"
#include "Topology/NeighborPointQuery.h"

#include "ParticleWriter.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLInstanceVisualModule.h>

#include <GLRenderEngine.h>

#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"

#include "StaticTriangularMesh.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-1.0f, 0.0f, 0.0f));
	scn->setUpperBound(Vec3f(1.0f, 1.0f, 1.0f));

	//Create a particle emitter
	auto emitter = scn->addNode(std::make_shared<SquareEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(0.0f, 0.5f, 0.5f));

	//Particle fluid node
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	emitter->connect(fluid->importParticleEmitters());

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->varPointSize()->setValue(0.002);
	ptRender->setColor(Vec3f(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	calculateNorm->outNorm()->connect(colorMapper->inScalar());

	colorMapper->outColor()->connect(ptRender->inColor());
	fluid->statePointSet()->connect(ptRender->inPointSet());

	fluid->graphicsPipeline()->pushModule(calculateNorm);
	fluid->graphicsPipeline()->pushModule(colorMapper);
	fluid->graphicsPipeline()->pushModule(ptRender);

	fluid->animationPipeline()->disable();

	//Barricade
	auto barricade = scn->addNode(std::make_shared<StaticTriangularMesh<DataType3f>>());
	barricade->varFileName()->setValue(getAssetPath() + "bowl/barricade.obj");
	barricade->varLocation()->setValue(Vec3f(0.1, 0.022, 0.5));

	auto sRenderf = std::make_shared<GLSurfaceVisualModule>();
	sRenderf->setColor(Vec3f(0.8, 0.52, 0.25));
	sRenderf->setVisible(true);
	sRenderf->varUseVertexNormal()->setValue(true);	// use generated smooth normal
	barricade->stateTopology()->connect(sRenderf->inTriangleSet());
	barricade->graphicsPipeline()->pushModule(sRenderf);

	//Scene boundary
	auto boundary = scn->addNode(std::make_shared<StaticTriangularMesh<DataType3f>>());
	boundary->varFileName()->setValue(getAssetPath() + "standard/standard_cube2.obj");
	boundary->graphicsPipeline()->disable();

	//SFI node
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalSFINode<DataType3f>>());
	auto pbd = std::make_shared<SemiAnalyticalPositionBasedFluidModel<DataType3f>>();
	pbd->varSmoothingLength()->setValue(0.0085);

	sfi->animationPipeline()->clear();
	sfi->stateTimeStep()->connect(pbd->inTimeStep());
	sfi->statePosition()->connect(pbd->inPosition());
	sfi->stateVelocity()->connect(pbd->inVelocity());
	sfi->stateForceDensity()->connect(pbd->inForce());
	sfi->stateTriangleVertex()->connect(pbd->inTriangleVertex());
	sfi->stateTriangleIndex()->connect(pbd->inTriangleIndex());
	sfi->animationPipeline()->pushModule(pbd);

	fluid->connect(sfi->importParticleSystems());
	barricade->connect(sfi->importBoundaryMeshs());
	boundary->connect(sfi->importBoundaryMeshs());

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