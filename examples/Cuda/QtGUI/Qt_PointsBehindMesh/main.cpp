#include <QtApp.h>
using namespace dyno;

#include "RigidBody/initializeRigidBody.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "ParticleSystem/ParticleFluid.h"
#include "initializeModeling.h"
#include "initializeIO.h"

#include "CubeModel.h"
#include "SphereModel.h"
#include "PlaneModel.h"
#include "Normal.h"
#include "PointsBehindMesh.h"
#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"

#include <ParticleSystem/SquareEmitter.h>
#include "ParticleSystem/MakeParticleSystem.h"
#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(15.5, 15.0, 15.5));
	scn->setLowerBound(Vec3f(-15.5, -15.0, -15.5));
	scn->setGravity(Vec3f(0.0f, -0.0f, 0.0f));

	//auto meshes = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	//meshes->varLocation()->setValue(Vec3f(0., 1.0, 0.));
	//meshes->varLatitude()->setValue(4);
	//meshes->varScale()->setValue(Vec3f(0.4, 0.4, 0.4));
	//meshes->varLongitude()->setValue(4);
	//auto meshes = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	
	auto meshes = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	meshes->varLocation()->setValue(Vec3f(0., 1.0, 0.));
	meshes->varScale()->setValue(Vec3f(0.4, 0.4, 0.4));

	auto pointset = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	pointset->varSamplingDistance()->setValue(0.005);
	pointset->varThickness()->setValue(0.045);
	meshes->stateTriangleSet()->connect(pointset->inTriangleSet());
	pointset->graphicsPipeline()->disable();


	auto relaxion = scn->addNode(std::make_shared<ParticleRelaxtionOnMesh<DataType3f>>());
	pointset->statePointNormal()->connect(relaxion->inParticleNormal());
	pointset->stateGhostPointSet()->connect(relaxion->inPointSet());
	pointset->outPointBelongTriangleIndex()->connect(relaxion->inParticleBelongTriangleIndex());
	meshes->stateTriangleSet()->connect(relaxion->inTriangleSet());

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

	relaxion->stateVelocity()->connect(calculateNorm->inVec());
	relaxion->statePointSet()->connect(ptRender->inPointSet());
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	colorMapper->outColor()->connect(ptRender->inColor());

	relaxion->graphicsPipeline()->pushModule(calculateNorm);
	relaxion->graphicsPipeline()->pushModule(colorMapper);
	relaxion->graphicsPipeline()->pushModule(ptRender);

	return scn;
}


int main()
{
	Modeling::initStaticPlugin();
	RigidBody::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	dynoIO::initStaticPlugin();


	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}