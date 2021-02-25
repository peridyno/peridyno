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

#include "Mapping/TriangleSetToTriangleSet.h"

#include "TexturedShape.h"
#include "AdjustStatus.h"

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

	std::shared_ptr<Node> root = scene.createNewScene<Node>();


	std::shared_ptr<HyperelasticBody<DataType3f>> elasticObj = std::make_shared<HyperelasticBody<DataType3f>>();

	elasticObj->loadVertexFromFile("../../data/pillow/Pillow.1");
 	elasticObj->scale(1);

	auto customModule = std::make_shared<AdjustStatus<DataType3f>>();
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
	elasticObj->setVisible(true);

	root->addChild(elasticObj);

	auto shape = std::make_shared<TexturedShape<DataType3f>>("shape1");
	shape->loadFile("../../data/pillow/PillowWithTex.obj");

	auto shapeRender = std::make_shared<SurfaceMeshRender>();
	shapeRender->setColor(Vector3f(1, 0, 1));
	shape->addVisualModule(shapeRender);

	elasticObj->addChild(shape);

	shape->setVisible(false);

	auto topoMapping = std::make_shared<TriangleSetToTriangleSet<DataType3f>>();

	topoMapping->setFrom(TypeInfo::cast<TetrahedronSet<DataType3f>>(elasticObj->getTopologyModule()));
	topoMapping->setTo(TypeInfo::cast<TriangleSet<DataType3f>>(shape->getTopologyModule()));

	topoMapping->setSearchingRadius(0.0075);

	elasticObj->addTopologyMapping(topoMapping);
	elasticObj->scale(0.1);

	shape->scale(0.1);
}


void CreateScene2()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setGravity(Vector3f(0.0f, -9.8f, 0.0f));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(-0.5, -0.2, -0.5), Vector3f(0.5, 0.8, 0.5), 0.05, true);
	root->varNormalFriction()->setValue(1.0f);
	root->varTangentialFriction()->setValue(0.5);


	std::shared_ptr<HyperelasticBody<DataType3f>> elasticObj = std::make_shared<HyperelasticBody<DataType3f>>();

	elasticObj->loadVertexFromFile("../../data/pillow/Pillow.1");
	elasticObj->scale(1);
//	elasticObj->varCollisionEnabled()->setValue(false);

// 	auto customModule = std::make_shared<AdjustStatus<DataType3f>>();
// 	elasticObj->currentPosition()->connect(customModule->inPosition());
// 	elasticObj->currentVelocity()->connect(customModule->inVelocity());
// 	elasticObj->currentAttribute()->connect(customModule->inVertexAttribute());
// 
// 	elasticObj->addCustomModule(customModule);

	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vector3f(0, 1, 1));
	elasticObj->addVisualModule(m_pointsRender);

	auto meshRender = std::make_shared<SurfaceMeshRender>();
	meshRender->setColor(Vector3f(1, 0, 1));
	elasticObj->addVisualModule(meshRender);
	elasticObj->setVisible(true);

	root->addChild(elasticObj);

	auto shape = std::make_shared<TexturedShape<DataType3f>>("shape1");
	shape->loadFile("../../data/pillow/PillowWithTex.obj");

	auto shapeRender = std::make_shared<SurfaceMeshRender>();
	shapeRender->setColor(Vector3f(1, 0, 1));
	shape->addVisualModule(shapeRender);

	elasticObj->addChild(shape);

	shape->setVisible(true);

	auto topoMapping = std::make_shared<TriangleSetToTriangleSet<DataType3f>>();

	topoMapping->setFrom(TypeInfo::cast<TetrahedronSet<DataType3f>>(elasticObj->getTopologyModule()));
	topoMapping->setTo(TypeInfo::cast<TriangleSet<DataType3f>>(shape->getTopologyModule()));

	topoMapping->setSearchingRadius(0.0075);

	elasticObj->addTopologyMapping(topoMapping);
	elasticObj->scale(0.1);

	shape->scale(0.1);

	root->addParticleSystem(elasticObj);
}

int main()
{
	CreateScene2();

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


