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
#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/HyperelasticityModule.h"
#include "ParticleSystem/HyperelasticityModule_test.h"
#include "SurfaceMeshRender.h"
#include "PointRenderModule.h"

#include "ManualControl.h"

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
	root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);

	std::shared_ptr<HyperelasticBody<DataType3f>> child3 = std::make_shared<HyperelasticBody<DataType3f>>();
	child3->setName("Hyperelastic Object");
	root->addParticleSystem(child3);

	child3->varHorizon()->setValue(0.007);

	auto ptRender1 = std::make_shared<PointRenderModule>();
	ptRender1->setColor(Vector3f(0, 1, 1));
	child3->addVisualModule(ptRender1);

	child3->loadVertexFromFile("../../data/smesh/iso_sphere.1");
	//	elasticObj->loadCentroidsFromFile("../../data/smesh/cube.1");
	child3->scale(0.1);
	child3->translate(Vector3f(0.5f, 0.2f, 0.5f));

	child3->setVisible(true);

	auto meshRender = std::make_shared<SurfaceMeshRender>();
	meshRender->setColor(Vector3f(1, 0, 1));
	child3->addVisualModule(meshRender);

	auto custom = std::make_shared<ManualControl<DataType3f>>();
	child3->currentPosition()->connect(custom->inPosition());
	child3->currentVelocity()->connect(custom->inVelocity());
	child3->currentAttribute()->connect(custom->inAttribute());
	//custom->addFixedPoint(0, Vector3f(0.5f, 0.5f, 0.5f));

	child3->addCustomModule(custom);

	auto module_hyper = std::dynamic_pointer_cast<HyperelasticityModule_test<DataType3f>>(child3->getElasticitySolver());
	module_hyper->setIterationNumber(1000);
	module_hyper->varisConvergeComputeField()->setValue(true);
	module_hyper->varconvergencyEpsilonField()->setValue(0.000001);
	module_hyper->setAlphaStepCompute(true);
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


