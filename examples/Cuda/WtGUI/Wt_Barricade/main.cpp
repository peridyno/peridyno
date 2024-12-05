#include "WtApp.h"

#include "SceneGraph.h"

#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"

#include "Topology/TriangleSet.h"
#include "Mapping/MergeTriangleSet.h"

#include "Collision/NeighborPointQuery.h"

#include "Module/CalculateNorm.h"
#include "BasicShapes/CubeModel.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLInstanceVisualModule.h>

#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
#include "SemiAnalyticalScheme/TriangularMeshBoundary.h"
#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"

#include "StaticTriangularMesh.h"

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-1.0f, 0.0f, 0.0f));
	scn->setUpperBound(Vec3f(1.0f, 1.0f, 1.0f));

	//auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());

	//Create a particle emitter
	auto emitter = scn->addNode(std::make_shared<SquareEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(0.0f, 0.5f, 0.5f));

	//Particle fluid node
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	emitter->connect(fluid->importParticleEmitters());

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->varPointSize()->setValue(0.002);
	ptRender->setColor(Color(1, 0, 0));
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

	//	fluid->animationPipeline()->disable();

	//Barricade
	auto barricade = scn->addNode(std::make_shared<StaticTriangularMesh<DataType3f>>());
	barricade->varFileName()->setValue(getAssetPath() + "bowl/barricade.obj");
	barricade->varLocation()->setValue(Vec3f(0.1, 0.022, 0.5));

	auto sRenderf = std::make_shared<GLSurfaceVisualModule>();
	sRenderf->setColor(Color(0.8f, 0.52f, 0.25f));
	sRenderf->setVisible(true);
	sRenderf->varUseVertexNormal()->setValue(true);	// use generated smooth normal
	barricade->stateTriangleSet()->connect(sRenderf->inTriangleSet());
	barricade->graphicsPipeline()->pushModule(sRenderf);

	//Scene boundary
	auto boundary = scn->addNode(std::make_shared<StaticTriangularMesh<DataType3f>>());
	boundary->varFileName()->setValue(getAssetPath() + "standard/standard_cube2.obj");
	boundary->graphicsPipeline()->disable();

	//SFI node
	auto sfi = scn->addNode(std::make_shared<TriangularMeshBoundary<DataType3f>>());
	auto pbd = std::make_shared<SemiAnalyticalPositionBasedFluidModel<DataType3f>>();
	pbd->varSmoothingLength()->setValue(0.0085);

	auto merge = scn->addNode(std::make_shared<MergeTriangleSet<DataType3f>>());
	boundary->stateTriangleSet()->connect(merge->inFirst());
	barricade->stateTriangleSet()->connect(merge->inSecond());

	fluid->connect(sfi->importParticleSystems());
	merge->stateTriangleSet()->connect(sfi->inTriangleSet());

	return scn;
}

int main(int argc, char** argv)
{
	WtApp app;

	app.setSceneGraphCreator(&createScene);
	app.setSceneGraph(createScene());

	app.mainLoop();

	return 0;
}