#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GlutGUI/GLApp.h"

#include "Framework/SceneGraph.h"
#include "Topology/PointSet.h"
#include "Framework/Log.h"

#include "ParticleSystem/ParticleElastoplasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "RigidBody/RigidBody.h"
#include "ParticleSystem/FractureModule.h"
#include "ParticleSystem/ElasticityModule.h"

#include "PointRenderModule.h"
#include "SurfaceMeshRender.h"

#include "ParticleSystem/HyperelastoplasticityBody.h"
#include "ParticleSystem/HyperelastoplasticityModule.h"

using namespace std;
using namespace dyno;

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setGravity(Vector3f(0.0f, -9.8f, 0.0f));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	// 	root->loadSDF("../data/bar/bar.sdf", false);
	// 	root->translate(Vector3f(0.2f, 0.2f, 0));
	//root->loadCube(Vector3f(0.4, 0.0, 0.4), Vector3f(0.6, 1.0, 0.6), 0.005, true);
	root->loadCube(Vector3f(0.0, 0.0, 0.0), Vector3f(1.0, 1.0, 1.0), 0.005, true);

	root->varNormalFriction()->setValue(1);
	root->varTangentialFriction()->setValue(1);

	std::shared_ptr<HyperelastoplasticityBody<DataType3f>> elasticObj = std::make_shared<HyperelastoplasticityBody<DataType3f>>();
	root->addParticleSystem(elasticObj);

	//elasticObj->loadVertexFromFile("../../data/smesh/flat_tet.1");
	elasticObj->loadVertexFromFile("../../data/smesh/iso_sphere.1");
	//elasticObj->loadVertexFromFile("../../data/smesh/test_collision");
	//elasticObj->loadVertexFromFile("../../data/smesh/cube.1");
	//elasticObj->loadCentroidsFromFile("../../data/smesh/cube.1");
 	elasticObj->scale(0.1);
 	elasticObj->translate(Vector3f(0.5f, 0.2f, 0.5f));

	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vector3f(0, 1, 1));
	elasticObj->addVisualModule(m_pointsRender);

	auto meshRender = std::make_shared<SurfaceMeshRender>();
	meshRender->setColor(Vector3f(1, 0, 1));
	elasticObj->addVisualModule(meshRender);
}


int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


