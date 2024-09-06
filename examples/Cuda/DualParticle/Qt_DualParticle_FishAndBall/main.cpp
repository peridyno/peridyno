//#include <GlfwApp.h>
#include <QtApp.h>
#include "SceneGraph.h"
#include <Log.h>
#include "ParticleSystem/StaticBoundary.h"
#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>
#include "DualParticleSystem/DualParticleFluidSystem.h"
#include "ParticleSystem/MakeParticleSystem.h"
#include <BasicShapes/SphereModel.h>
#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>
#include <StaticTriangularMesh.h>
#include <GLSurfaceVisualModule.h>
#include "Auxiliary/DataSource.h"
#include "PointsLoader.h"
using namespace std;
using namespace dyno;

bool useVTK = false;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(3.0, 3, 3.0));
	scn->setLowerBound(Vec3f(-3.0, -3.0, -3.0));

	auto ptsLoader = scn->addNode(std::make_shared<PointsLoader<DataType3f>>());
	ptsLoader->varFileName()->setValue(getAssetPath() + "fish/FishPoints.obj");
	ptsLoader->varRotation()->setValue(Vec3f(0.0f, 3.14 * 2 / 5, 0.0f));
	ptsLoader->varLocation()->setValue(Vec3f(0.0f, 0.4f, 0.30f));
	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f >>());
	ptsLoader->outPointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<DualParticleFluidSystem<DataType3f>>());
	fluid->varReshuffleParticles()->setValue(true);
	initialParticles->connect(fluid->importInitialStates());

	//Create a boundary
	auto boundary = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>()); ;
	boundary->loadCube(Vec3f(-0.5, -2.0, -0.5), Vec3f(1.5, 2, 1.5), 0.02, true);
	//boundary->loadSDF(getAssetPath() + "bowl/bowl.sdf", false);
	fluid->connect(boundary->importParticleSystems());

	auto ball = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	ball->varScale()->setValue(Vec3f(0.38));
	ball->varLocation()->setValue(Vec3f(0.0, 0.0, 0.3));
	auto sRenderf = std::make_shared<GLSurfaceVisualModule>();
	sRenderf->setColor(Color(0.8f, 0.52f, 0.25f));
	sRenderf->setVisible(true);
	sRenderf->varUseVertexNormal()->setValue(true);	// use generated smooth normal
	ball->stateTriangleSet()->connect(sRenderf->inTriangleSet());
	ball->graphicsPipeline()->pushModule(sRenderf);

	auto pm_collide = scn->addNode(std::make_shared <TriangularMeshBoundary<DataType3f >>());
	ball->stateTriangleSet()->connect(pm_collide->inTriangleSet());
	fluid->connect(pm_collide->importParticleSystems());
	//fluid->stateVelocity()->connect(pm_collide->inVelocity());

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
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
	vpRender->varPointSize()->setValue(0.0005);
	fluid->graphicsPipeline()->pushModule(vpRender);

	return scn;
}

int main()
{

	//GlfwApp window;
	QtApp window;
	window.setSceneGraph(createScene());
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}


