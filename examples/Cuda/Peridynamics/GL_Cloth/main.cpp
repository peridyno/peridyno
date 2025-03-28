#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Peridynamics/ElasticBody.h>
#include <Peridynamics/Cloth.h>

#include <BasicShapes/SphereModel.h>
#include <Volume/BasicShapeToVolume.h>
#include <Multiphysics/VolumeBoundary.h>

#include <BasicShapes/PlaneModel.h>

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

	auto sphereModel = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	sphereModel->varLocation()->setValue(Vec3f(0.0, 0.7f, 0.0));
	sphereModel->varRadius()->setValue(0.2f);

	auto sphere2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	sphere2vol->varGridSpacing()->setValue(0.05f);
	sphereModel->connect(sphere2vol->importShape());

	auto cloth = scn->addNode(std::make_shared<Cloth<DataType3f>>());
	cloth->setDt(0.001f);
	plane->stateTriangleSet()->connect(cloth->inTriangleSet());
	//cloth->loadSurface(getAssetPath() + "cloth_shell/mesh_80_80.obj");

	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	sphere2vol->connect(boundary->importVolumes());

	cloth->connect(boundary->importTriangularSystems());

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