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

#include "VolumeBoundary.h"
#include "FixBoundary.h"

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setGravity(Vector3f(0.0f, -1.0, 0.0f));

	std::shared_ptr<VolumeBoundary<DataType3f>> submarine = scene.createNewScene<VolumeBoundary<DataType3f>>();
	auto cubeSurface = submarine->loadSDF("../../data/submarine/submarine.sdf", false);

	auto triTopo = TypeInfo::cast<TriangleSet<DataType3f>>(cubeSurface->getTopologyModule());
	triTopo->loadObjFile("../../data/submarine/submarine.obj");

	cubeSurface->addVisualModule(std::make_shared<SurfaceMeshRender>());
	cubeSurface->setVisible(true);

	std::shared_ptr<HyperelastoplasticityBody<DataType3f>> elasticObj = std::make_shared<HyperelastoplasticityBody<DataType3f>>();
	elasticObj->loadVertexFromFile("../../data/submarine/ice.1");
	//elasticObj->loadVertexFromFile("../../data/submarine/ice_small.1");
	elasticObj->varCollisionEnabled()->setValue(true);

	auto customModule = std::make_shared<FixBoundary<DataType3f>>();
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
	
	submarine->addParticleSystem(elasticObj);
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


