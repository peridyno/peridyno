#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GlutGUI/GLApp.h"

#include "Framework/SceneGraph.h"
#include "Topology/PointSet.h"
#include "Topology/TetrahedronSet.h"
#include "Framework/Log.h"

#include "ParticleSystem/HyperelasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/HyperelasticityModule.h"
#include "ParticleSystem/HyperelasticityModule_test.h"
#include "SurfaceMeshRender.h"
#include "PointRenderModule.h"


using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setGravity(Vector3f(0.0f, -9.8f, 0.0f));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);

	std::shared_ptr<HyperelasticBody<DataType3f>> elasticObj = std::make_shared<HyperelasticBody<DataType3f>>();
	root->addParticleSystem(elasticObj);

	//elasticObj->varHorizon()->setValue(0.0001);

	auto pointRender = std::make_shared<PointRenderModule>();
	pointRender->setColor(Vector3f(0, 1, 1));
	elasticObj->addVisualModule(pointRender);

	elasticObj->setMass(1.0);

	Vector3f center(0.0, 0.0, 0.0);
	Vector3f rectangle(0.06, 0.05, 0.05);
	//elasticObj->loadParticles(center- rectangle, center + rectangle, 0.005);
	elasticObj->loadVertexFromFile("../../data/smesh/cube.1");
//	elasticObj->loadCentroidsFromFile("../../data/smesh/cube.1");
 	elasticObj->scale(0.1);
 	elasticObj->translate(Vector3f(0.5f, 0.2f, 0.5f));

//	elasticObj->loadStandardTet();
	//elasticObj->loadStandardSimplex();


	double x_border = 0.5;
	elasticObj->setVisible(true);

	//elasticObj->scale(0.1);

	//elasticObj->getMeshNode()->setActive(false);

	auto meshRender = std::make_shared<SurfaceMeshRender>();
	meshRender->setColor(Vector3f(1, 0, 1));
	elasticObj->addVisualModule(meshRender);
// 	elasticObj->getMeshNode()->addVisualModule(meshRender);
// 
// 	elasticObj->getMeshNode()->setVisible(true);
}


int main()
{
	CreateScene();

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


