#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Peridynamics/ElasticBody.h>
#include <Peridynamics/Cloth.h>

#include <Multiphysics/VolumeBoundary.h>

#include <PlaneModel.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLWireframeVisualModule.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varSegmentX()->setValue(80);
	plane->varSegmentZ()->setValue(80);
	plane->varLocation()->setValue(Vec3f(0.0f, 0.9f, 0.0f));
	plane->graphicsPipeline()->disable();

	auto cloth = scn->addNode(std::make_shared<Cloth<DataType3f>>());
	cloth->setDt(0.001f);
	plane->stateTriangleSet()->connect(cloth->inTriangleSet());
	//cloth->loadSurface(getAssetPath() + "cloth_shell/mesh_80_80.obj");

	auto root = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	root->loadShpere(Vec3f(0.0, 0.7f, 0.0), 0.08f, 0.005f, false, true);

	cloth->connect(root->importTriangularSystems());

	auto pointRenderer = std::make_shared<GLPointVisualModule>();
	pointRenderer->setColor(Color(1, 0.2, 1));
	pointRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
	pointRenderer->varPointSize()->setValue(0.002f);
	cloth->stateTriangleSet()->connect(pointRenderer->inPointSet());
	cloth->stateVelocity()->connect(pointRenderer->inColor());

	cloth->graphicsPipeline()->pushModule(pointRenderer);
	cloth->setVisible(true);

	auto wireRenderer = std::make_shared<GLWireframeVisualModule>();
	wireRenderer->varBaseColor()->setValue(Color(1.0, 0.8, 0.8));
	wireRenderer->varRadius()->setValue(0.001f);
	wireRenderer->varRenderMode()->setCurrentKey(GLWireframeVisualModule::CYLINDER);
	cloth->stateTriangleSet()->connect(wireRenderer->inEdgeSet());
	cloth->graphicsPipeline()->pushModule(wireRenderer);

	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
	cloth->stateTriangleSet()->connect(surfaceRenderer->inTriangleSet());
	cloth->graphicsPipeline()->pushModule(surfaceRenderer);
	//cloth->getSurface()->graphicsPipeline()->pushPersistentModule(surfaceRenderer);

	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}