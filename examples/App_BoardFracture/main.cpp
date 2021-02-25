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

#include "ParticleSystem/HyperelasticFractureModule.h"

#include "VolumeBoundary.h"
#include "AdjustBoard.h"

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
	scene.setGravity(Vector3f(0.0f, 0.0f, 0.0f));

	std::shared_ptr<VolumeBoundary<DataType3f>> ball = scene.createNewScene<VolumeBoundary<DataType3f>>();
	auto cubeSurface = ball->loadSDF("../../data/board/ball.sdf", false);

	cubeSurface->addVisualModule(std::make_shared<SurfaceMeshRender>());
	cubeSurface->setVisible(true);

	auto triTopo = TypeInfo::cast<TriangleSet<DataType3f>>(cubeSurface->getTopologyModule());
	triTopo->loadObjFile("../../data/board/ball.obj");

	ball->translate(Vector3f(0.8f, 0.0f, 0.0f));


	std::shared_ptr<HyperelastoplasticityBody<DataType3f>> elasticObj = std::make_shared<HyperelastoplasticityBody<DataType3f>>();

	elasticObj->loadVertexFromFile("../../data/board/board.1");
 	elasticObj->scale(1);
	elasticObj->varCollisionEnabled()->setValue(false);
	elasticObj->getFractureModule()->varCriticalStretch()->setValue(1.1);

	auto customModule = std::make_shared<AdjustBoard<DataType3f>>();
	elasticObj->currentPosition()->connect(customModule->inPosition());
	elasticObj->currentVelocity()->connect(customModule->inVelocity());
	elasticObj->currentAttribute()->connect(customModule->inVertexAttribute());

	elasticObj->addCustomModule(customModule);

	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vector3f(0, 1, 1));
	elasticObj->addVisualModule(m_pointsRender);

	auto meshRender = std::make_shared<SurfaceMeshRender>();
	meshRender->setColor(Vector3f(1, 0, 1));
	elasticObj->addVisualModule(meshRender);

	ball->addParticleSystem(elasticObj);
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


