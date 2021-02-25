#include <iostream>
#include "GlutGUI/GLApp.h"
#include "Framework/SceneGraph.h"


#include "SurfaceMeshRender.h"
#include "ElementRender.h"
#include "PointRenderModule.h"
#include "Topology/TriangleSet.h"
#include "Topology/DiscreteElements.h"
#include "Topology/Primitive3D.h"
#include "ParticleSystem/TriangularSurfaceMeshNode.h"

#include "RigidBody/RigidBodySystem.h"


using namespace dyno;


//typedef typename TOrientedBox3D<Real> Box3D;

void test_render()
{
	Log::sendMessage(Log::Info, "Simulation start");

	SceneGraph& scene = SceneGraph::getInstance();
	scene.setFrameRate(30);

	scene.setUpperBound(Vector3f(1.2, 1.2, 1.2));
	scene.setLowerBound(Vector3f(-0.2, -0.2, -0.2));

	std::shared_ptr<RigidBodySystem<DataType3f>> root = scene.createNewScene<RigidBodySystem<DataType3f>>();


	std::shared_ptr<DiscreteElements<DataType3f>> DE = std::make_shared<DiscreteElements<DataType3f>>();

	
	
	//DE->addBox(Box3D(Vector3f(0.35, 0.4, 0.5), Vector3f(1.0, 0.0, 0.0), Vector3f(0.0, 1.0, 0.0), Vector3f(0.0, 0.0, 1.0), Vector3f(0.15, 0.1, 0.1)));
	for(int i = 0; i < 5; i ++)
		DE->addBox(Box3D(Vector3f(0.2 + 0.03 * i, 0.1 + 0.15 * i, 0.5), Vector3f(1.0, 0.0, 0.0), Vector3f(0.0, 1.0, 0.0), Vector3f(0.0, 0.0, 1.0), Vector3f(0.1, 0.065, 0.065)));
		
		
	//DE->addBox(Box3D(Vector3f(0.65, 0.5, 0.5), Vector3f(1.0, 0.0, 0.0), Vector3f(0.0, 1.0, 0.0), Vector3f(0.0, 0.0, 1.0), Vector3f(0.15, 0.1, 0.1)));
	//DE->addBox(Box3D(Vector3f(0.65, 0.1, 0.5), Vector3f(1.0, 0.0, 0.0), Vector3f(0.0, 1.0, 0.0), Vector3f(0.0, 0.0, 1.0), Vector3f(0.15, 0.1, 0.1)));



	DE->initialize();
	
	root->setTopologyModule(DE);
	

	auto sRender = std::make_shared<ElementRender>();
	sRender->setColor(Vector3f(1, 1, 0));


	root->addVisualModule(sRender);

	
	GLApp window;
	window.createWindow(1024, 768);
	window.mainLoop();
}

int main()
{
	
	//creat_scene_fluid();
	test_render();
	return 0;
}


