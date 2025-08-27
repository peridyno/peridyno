#include <QtApp.h>
#include "Peridynamics/Cloth.h"
#include <SceneGraph.h>
#include <Volume/VolumeLoader.h>
#include <Volume/VolumeGenerator.h>
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
	object->varFileName()->setValue(getAssetPath() + "cloth_shell/model_ball.obj");
	
	auto volGenerator = scn->addNode(std::make_shared<VolumeGenerator<DataType3f>>());
	volGenerator->varSpacing()->setValue(0.01f);
	object->stateTriangleSet()->connect(volGenerator->inTriangleSet());

// 	auto volLoader = scn->addNode(std::make_shared<VolumeLoader<DataType3f>>());
// 	volLoader->varFileName()->setValue(getAssetPath() + "cloth_shell/model_sdf.sdf");

	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	volGenerator->connect(boundary->importVolumes());

	auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>());
	cloth->loadSurface(getAssetPath() + "cloth_shell/mesh_120.obj");
	cloth->connect(boundary->importTriangularSystems()); 
	cloth->setDt(0.001f);
	{
		auto solver = cloth->animationPipeline()->findFirstModule<CoSemiImplicitHyperelasticitySolver<DataType3f>>();

		solver->setGrad_res_eps(1e-4);
		solver->varIterationNumber()->setValue(10);
		solver->setAccelerated(true);
	}

	auto surfaceRendererCloth = std::make_shared<GLSurfaceVisualModule>();
	surfaceRendererCloth->setColor(Color(0.4,0.4,1.0));

	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
	surfaceRenderer->setColor(Color(0.4,0.4,0.4));
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
	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}