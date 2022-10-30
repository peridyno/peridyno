#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Topology/PointSet.h>

#include <ParticleSystem/StaticBoundary.h>

#include <Peridynamics/ElastoplasticBody.h>
#include <Peridynamics/ElasticBody.h>


#include <RigidBody/RigidBody.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>

#include <Node/GLSurfaceVisualNode.h>

#include <Mapping/PointSetToTriangleSet.h>

#include <SurfaceMeshLoader.h>

using namespace std;
using namespace dyno;

//TODO: this example crashes for some unknown reasons.

std::shared_ptr<SceneGraph> createScene()
{
	//Create a scene graph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a elastoplastic object
	auto elastoplasticBody = scn->addNode(std::make_shared<ElastoplasticBody<DataType3f>>());	

	elastoplasticBody->setVisible(false);
  	elastoplasticBody->loadParticles(Vec3f(-1.1), Vec3f(1.15), 0.1);
	elastoplasticBody->scale(0.05);
	elastoplasticBody->translate(Vec3f(0.3, 0.2, 0.5));

	//Create a surface mesh loader
	auto surfaceMeshLoader = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
	surfaceMeshLoader->varFileName()->setValue(getAssetPath() + "standard/standard_cube20.obj");
	surfaceMeshLoader->varScale()->setValue(Vec3f(0.05f));
	surfaceMeshLoader->varLocation()->setValue(Vec3f(0.3, 0.2, 0.5));

	//Create a topology mapper
	auto topoMapper = scn->addNode(std::make_shared<PointSetToTriangleSet<DataType3f>>());

	auto outTop = elastoplasticBody->statePointSet()->promoteOuput();
 	outTop->connect(topoMapper->inPointSet());
	surfaceMeshLoader->outTriangleSet()->connect(topoMapper->inInitialShape());

	auto surfaceVisualizer = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	topoMapper->outShape()->connect(surfaceVisualizer->inTriangleSet());

	//Create a elastic object
	auto elasticBody = scn->addNode(std::make_shared<ElasticBody<DataType3f>>());
	elasticBody->setVisible(false);
	elasticBody->loadParticles(Vec3f(-1.1), Vec3f(1.15), 0.1);
	elasticBody->scale(0.05);
	elasticBody->translate(Vec3f(0.5, 0.2, 0.5));

	auto surfaceMeshLoader2 = scn->addNode(std::make_shared<SurfaceMeshLoader<DataType3f>>());
	surfaceMeshLoader2->varFileName()->setValue(getAssetPath() + "standard/standard_cube20.obj");
	surfaceMeshLoader2->varScale()->setValue(Vec3f(0.05f));
	surfaceMeshLoader2->varLocation()->setValue(Vec3f(0.5, 0.2, 0.5));

	//Create a topology mapper
	auto topoMapper2 = scn->addNode(std::make_shared<PointSetToTriangleSet<DataType3f>>());

	auto outTop2 = elasticBody->statePointSet()->promoteOuput();
	outTop2->connect(topoMapper2->inPointSet());
	surfaceMeshLoader2->outTriangleSet()->connect(topoMapper2->inInitialShape());

	auto surfaceVisualizer2 = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	topoMapper2->outShape()->connect(surfaceVisualizer2->inTriangleSet());

	auto boundary = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	boundary->loadCube(Vec3f(0), Vec3f(1), 0.005, true);
	elastoplasticBody->connect(boundary->importParticleSystems());
	elasticBody->connect(boundary->importParticleSystems());

	return scn;
}

int main()
{
	GlfwApp window;
	window.setSceneGraph(createScene());
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}


