#include <QtApp.h>
using namespace dyno;

#include "RigidBody/initializeRigidBody.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "ParticleSystem/ParticleFluid.h"
#include "initializeModeling.h"
#include "initializeIO.h"

#include "BasicShapes/CubeModel.h"
#include "BasicShapes/SphereModel.h"
#include "BasicShapes/PlaneModel.h"
#include "Normal.h"
#include "Commands/PointsBehindMesh.h"
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

	auto meshes_1 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	meshes_1->varLocation()->setValue(Vec3f(0., 1.0, 0.));
	meshes_1->varLatitude()->setValue(8);
	meshes_1->varScale()->setValue(Vec3f(0.6, 0.6, 0.6));
	meshes_1->varLongitude()->setValue(8);

	auto pointset_1 = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	pointset_1->varSamplingDistance()->setValue(0.005);
	pointset_1->varThickness()->setValue(0.045);
	meshes_1->stateTriangleSet()->connect(pointset_1->inTriangleSet());
	//pointset0->graphicsPipeline()->disable();

	auto meshes_2 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	meshes_2->varLocation()->setValue(Vec3f(1., 1.0, 0.));
	meshes_2->varScale()->setValue(Vec3f(0.4, 0.4, 0.4));

	auto pointset_2 = scn->addNode(std::make_shared<ParticleRelaxtionOnMesh<DataType3f>>());
	pointset_2->varSamplingDistance()->setValue(0.005);
	pointset_2->varThickness()->setValue(0.045);
	meshes_2->stateTriangleSet()->connect(pointset_2->inTriangleSet());
	pointset_2->graphicsPipeline()->clear();


	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

	pointset_2->stateVelocity()->connect(calculateNorm->inVec());
	pointset_2->stateGhostPointSet()->connect(ptRender->inPointSet());
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	colorMapper->outColor()->connect(ptRender->inColor());

	pointset_2->graphicsPipeline()->pushModule(calculateNorm);
	pointset_2->graphicsPipeline()->pushModule(colorMapper);
	pointset_2->graphicsPipeline()->pushModule(ptRender);

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