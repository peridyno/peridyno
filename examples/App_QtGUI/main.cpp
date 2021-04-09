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
#include "ParticleSystem/ParticleWriter.h"

#include "Framework/SceneGraph.h"
#include "Framework/ControllerAnimation.h"

#include "Topology/TriangleSet.h"

#include "RigidBody/RigidBody.h"


using namespace std;
using namespace dyno;


int main()
{
//	creare_scene_init();

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