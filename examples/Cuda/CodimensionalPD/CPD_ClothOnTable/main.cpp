#include <UbiApp.h>
#include "Peridynamics/Cloth.h"
#include <SceneGraph.h>

#include <Volume/VolumeLoader.h>
#include <Multiphysics/VolumeBoundary.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include "Peridynamics/CodimensionalPD.h"

#include "StaticMeshLoader.h"
using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-1.5, 0, -1.5));
	scn->setUpperBound(Vec3f(1.5, 3, 1.5));
	auto object = scn->addNode(std::make_shared<StaticMeshLoader<DataType3f>>());
	object->varFileName()->setValue(getAssetPath() + "cloth_shell/table/table.obj");

	auto volLoader = scn->addNode(std::make_shared<VolumeLoader<DataType3f>>());
	volLoader->varFileName()->setValue(getAssetPath() + "cloth_shell/table/table.sdf");

	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	volLoader->connect(boundary->importVolumes());

	auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>());
	//also try:
	//auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>(0.3, 8000, 0.03,7e-4));
	//auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>(0.3, 8000, 0.3,7e-4));
	//auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>(0.3, 8000, 0.0,7e-4));
	cloth->loadSurface(getAssetPath() + "cloth_shell/mesh40k_1_h90.obj");
	cloth->connect(boundary->importTriangularSystems());

	auto surfaceRendererCloth = std::make_shared<GLSurfaceVisualModule>();
	surfaceRendererCloth->setColor(Color(0.4, 0.4, 1.0));

	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
	surfaceRenderer->setColor(Color(0.8, 0.8, 0.8));
	surfaceRenderer->varUseVertexNormal()->setValue(true);
	cloth->stateTriangleSet()->connect(surfaceRendererCloth->inTriangleSet());
	object->stateTriangleSet()->connect(surfaceRenderer->inTriangleSet());
	cloth->graphicsPipeline()->pushModule(surfaceRendererCloth);
	object->graphicsPipeline()->pushModule(surfaceRenderer);
	cloth->setVisible(true);
	object->setVisible(true);
	scn->printNodeInfo(true);
	scn->printSimulationInfo(true);

	return scn;
}

int main()
{
	UbiApp window(GUIType::GUI_QT);
	window.setSceneGraph(createScene());

	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}