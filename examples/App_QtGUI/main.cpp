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
#include "ParticleSystem/SemiAnalyticalSFINode.h"
#include "ParticleSystem/TriangularSurfaceMeshNode.h"

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

void create_scene_semianylitical()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1.2));
	scene.setLowerBound(Vector3f(-0.2));

	scene.setFrameRate(1000);

	std::shared_ptr<SemiAnalyticalSFINode<DataType3f>> root = scene.createNewScene<SemiAnalyticalSFINode<DataType3f>>();
	root->setName("SemiAnalyticalSFI");
	//root->loadMesh();

	auto writer = std::make_shared<ParticleWriter<DataType3f>>();
	writer->setNamePrefix("particles_");
	root->getParticlePosition()->connect(&writer->m_position);
	root->getParticleMass()->connect(&writer->m_color_mapping);
	//root->addModule(writer);


	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(child1);
	child1->setName("fluid");
	//child1->loadParticles(Vector3f(0.75, 0.05, 0.75), Vector3f(0.85, 0.35, 0.85), 0.005);
	child1->setMass(1);

	std::shared_ptr<ParticleEmitterSquare<DataType3f>> child3 = std::make_shared<ParticleEmitterSquare<DataType3f>>();
	child1->addParticleEmitter(child3);


	auto pRenderer = std::make_shared<PVTKPointSetRender>();
	pRenderer->setName("VTK Point Renderer");


	//auto pRenderer = std::make_shared<PointRenderModule>();
	//pRenderer->setColor(Vector3f(0, 0, 1));

	child1->addVisualModule(pRenderer);


	std::shared_ptr <TriangularSurfaceMeshNode<DataType3f>> child2 = std::make_shared<TriangularSurfaceMeshNode<DataType3f>>("boundary");
	child2->getTriangleSet()->loadObjFile("../../data/standard/standard_cube_01.obj");

	root->addTriangularSurfaceMeshNode(child2);


	QtApp window;
	window.createWindow(1024, 768);
	//printf("outside 4\n");
	auto classMap = Object::getClassMap();

	for (auto const c : *classMap)
		std::cout << "Class Name: " << c.first << std::endl;

	window.mainLoop();
}

int main()
{
	auto& v = creare_scene_init();
	v.resize(5);

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