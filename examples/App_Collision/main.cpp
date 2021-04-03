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

#include "PointRenderModule.h"

#include "ParticleSystem/PositionBasedFluidModel.h"
#include "ParticleSystem/Peridynamics.h"

#include "Collision/CollidableSDF.h"
#include "Collision/CollidablePoints.h"
#include "Collision/CollisionSDF.h"
#include "ParticleSystem/FixedPoints.h"
#include "Collision/CollisionPoints.h"
#include "ParticleSystem/ParticleSystem.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/ElasticityModule.h"
#include "ParticleSystem/ParticleElastoplasticBody.h"
#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/SolidFluidInteraction.h"
#include "Mapping/PointSetToPointSet.h"
#include "SurfaceMeshRender.h"

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setFrameRate(500);
	scene.setUpperBound(Vector3f(1, 2.0, 1));
	scene.setLowerBound(Vector3f(0, 0.0, 0));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0, 0.0, 0), Vector3f(1, 2.0, 1), 0.015f, true);
	//root->loadSDF("box.sdf", true);

	std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
	// 

	root->addChild(sfi);
	sfi->setInteractionDistance(0.03);

	for (int i = 0; i < 6; i++)
	{
		std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
		root->addParticleSystem(bunny);

		auto sRender = std::make_shared<SurfaceMeshRender>();
		bunny->getSurfaceNode()->addVisualModule(sRender);
		
		if (i % 2 == 0)
		{
			sRender->setColor(Vector3f(1, 1, 0));
		}
		else
			sRender->setColor(Vector3f(1, 0, 1));
		
		bunny->varHorizon()->setValue(0.03f);
		bunny->setMass(1.0);
		bunny->loadParticles("../../data/bunny/sparse_bunny_points.obj");
		bunny->loadSurface("../../data/bunny/sparse_bunny_mesh.obj");
		bunny->translate(Vector3f(0.4, 0.2 + i * 0.3, 0.8));
		bunny->setVisible(false);
		bunny->getElasticitySolver()->setIterationNumber(10);
		//bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
		bunny->getTopologyMapping()->setSearchingRadius(0.05);

		sfi->addParticleSystem(bunny);
	}
}

int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


