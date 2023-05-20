#include <GlfwApp.h>
#include "Peridynamics/Cloth.h"
#include <SceneGraph.h>
#include <Log.h>
#include <Multiphysics/VolumeBoundary.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include "Peridynamics/CodimensionalPD.h"
#include "StaticTriangularMesh.h"
using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-1.5, 0, -1.5));
	scn->setUpperBound(Vec3f(1.5, 3, 1.5)); 
	
	auto object = scn->addNode(std::make_shared<StaticTriangularMesh<DataType3f>>());
	object->varFileName()->setValue(getAssetPath() + "cloth_shell/model_ball.obj");
	

	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	boundary->loadCube(Vec3f(-1.5,0,-1.5), Vec3f(1.5,3,1.5), 0.005f, true);
	//boundary->loadShpere(Vec3f(0.5, 0.6f, 0.5), 0.15f, 0.005f, false, true); 
	boundary->loadSDF(getAssetPath() + "cloth_shell/model_sdf.sdf");

	auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>(0.15f,2e1f,0.0f));
	cloth->loadSurface(getAssetPath() + "cloth_shell/mesh_120.obj");
	cloth->connect(boundary->importTriangularSystems()); 
	cloth->setDt(0.001f);
	cloth->setGrad_ite_eps(1e-4);
	cloth->setMaxIteNumber(10);
	cloth->setAccelerated(true);
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
	scn->printModuleInfo(true);

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