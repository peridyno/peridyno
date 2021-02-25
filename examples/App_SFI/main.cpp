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
#include "Framework/Gravity.h"
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
#include "PointRenderModule.h"

using namespace std;
using namespace dyno;

void CreateScene()
{
	printf("0\n");
	SceneGraph& scene = SceneGraph::getInstance();
//	scene.setUpperBound(Vector3f(1, 1.0, 0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();

	printf("0.1\n");

	root->loadCube(Vector3f(0), Vector3f(1), 0.015f, true);
	printf("1\n");
//  
	std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(fluid);
	printf("11\n");
	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vector3f(0, 0, 1));
	ptRender->setColorRange(0, 1);
	fluid->addVisualModule(ptRender);
	printf("111\n");
	//fluid->loadParticles("../data/fluid/fluid_point.obj");
	fluid->loadParticles(Vector3f(0), Vector3f(0.5, 1.0, 1.0), 0.015f);
	fluid->setMass(10);
	//fluid->getVelocity()->connect(fluid->getRenderModule()->m_vecIndex);
	printf("1111\n");
	std::shared_ptr<PositionBasedFluidModel<DataType3f>> pbd = std::make_shared<PositionBasedFluidModel<DataType3f>>();
	fluid->currentPosition()->connect(&pbd->m_position);
	fluid->currentVelocity()->connect(&pbd->m_velocity);
	fluid->currentForce()->connect(&pbd->m_forceDensity);
	pbd->setSmoothingLength(0.02);

	fluid->setNumericalModel(pbd);

	printf("111111\n");
	std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
	// 
	sfi->setInteractionDistance(0.02);
	root->addChild(sfi);

	for (int i = 0; i < 3; i++)
	{
		printf("%d\n", i);
		std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
		root->addParticleSystem(bunny);
		bunny->setMass(1.0);
		bunny->loadParticles("../../data/bunny/sparse_bunny_points.obj");
		bunny->loadSurface("../../data/bunny/sparse_bunny_mesh.obj");
		bunny->translate(Vector3f(0.75, 0.2, 0.4 + i * 0.3));
		bunny->setVisible(false);
		bunny->getElasticitySolver()->setIterationNumber(10);
		bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
		bunny->getTopologyMapping()->setSearchingRadius(0.05);

		auto sRender = std::make_shared<SurfaceMeshRender>();
		bunny->getSurfaceNode()->addVisualModule(sRender);
		sRender->setColor(Vector3f(i*0.3f, 1 - i*0.3f, 1.0));

		sfi->addParticleSystem(bunny);
	}


	sfi->addParticleSystem(fluid);
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


