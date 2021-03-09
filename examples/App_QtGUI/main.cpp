#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "QtGUI/QtApp.h"
#include "QtGUI/PVTKSurfaceMeshRender.h"
#include "QtGUI/PVTKPointSetRender.h"

#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/ElasticityModule.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/ParticleEmitter.h"
#include "ParticleSystem/ParticleEmitterRound.h"
#include "ParticleSystem/ParticleEmitterSquare.h"
#include "ParticleSystem/StaticMeshBoundary.h"
#include "ParticleSystem/ParticleWriter.h"
#include "ParticleSystem/StaticMeshBoundary.h"

#include "Framework/SceneGraph.h"
#include "Framework/ControllerAnimation.h"

#include "Topology/TriangleSet.h"

#include "RigidBody/RigidBody.h"


using namespace std;
using namespace dyno;

std::vector<float> test_vector;

std::vector<float>& creare_scene_init()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1.5, 1.5, 1.5));
	scene.setLowerBound(Vector3f(-1.5, -0.5, -1.5));

	std::shared_ptr<StaticMeshBoundary<DataType3f>> root = scene.createNewScene<StaticMeshBoundary<DataType3f>>();
	root->loadMesh("../../data/bowl/b3.obj");
	//root->loadMesh("../../data/standard/standard_cube_01.obj"); // bug

	root->setName("StaticMesh");
	//root->loadMesh();
	

	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(child1);
	child1->setName("fluid");

 	std::shared_ptr<ParticleEmitterSquare<DataType3f>> child2 = std::make_shared<ParticleEmitterSquare<DataType3f>>();
	child1->addParticleEmitter(child2);
	child1->setMass(100);

	auto pRenderer = std::make_shared<PVTKPointSetRender>();
	pRenderer->setName("VTK Point Renderer");
	child1->addVisualModule(pRenderer);
    printf("outside visual\n");
// 	printf("outside 1\n");
// 	
	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	
	//rigidbody->loadShape("../../data/bowl/b3.obj");

	//rigidbody->loadShape("../../data/standard/standard_cube_01.obj");

	printf("outside 2\n");
	auto sRenderer = std::make_shared<PVTKSurfaceMeshRender>();
	sRenderer->setName("VTK Surface Renderer");
	rigidbody->getSurface()->addVisualModule(sRenderer);
	rigidbody->setActive(false);

	root->addRigidBody(rigidbody);

	SceneGraph::Iterator it_end(nullptr);
	for (auto it = scene.begin(); it != it_end; it++)
	{
		auto node_ptr = it.get();
		std::cout << node_ptr->getClassInfo()->getClassName() << ": " << node_ptr.use_count() << std::endl;
	}


	std::cout << "Rigidbody use count: " << rigidbody.use_count() << std::endl;

//	std::cout << "Rigidbody use count: " << rigidbody.use_count() << std::endl;
	test_vector.resize(10);
	return test_vector;
}

int main()
{
	creare_scene_init();

	printf("outside 3\n");
	QtApp window;
	window.createWindow(1024, 768);
	printf("outside 4\n");
	auto classMap = Object::getClassMap();

	// 	for (auto const c : *classMap)
	// 		std::cout << "Class Name: " << c.first << std::endl;

	window.mainLoop();

//	std::cout << "Rigidbody use count: " << SceneGraph::getInstance().getRootNode()->getChildren().front().use_count() << std::endl;

	//create_scene_semianylitical();
	return 0;
}