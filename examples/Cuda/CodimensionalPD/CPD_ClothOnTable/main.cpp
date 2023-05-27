#include <GlfwApp.h>
#include "Peridynamics/Cloth.h"
#include <SceneGraph.h>
#include <Log.h>
#include <Multiphysics/VolumeBoundary.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include "Peridynamics/CodimensionalPD.h"
#include "../Modeling/StaticTriangularMesh.h"
using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-1.5, 0, -1.5));
	scn->setUpperBound(Vec3f(1.5, 3, 1.5));
	auto object = scn->addNode(std::make_shared<StaticTriangularMesh<DataType3f>>());
	object->varFileName()->setValue(getAssetPath() + "cloth_shell/table/table.obj");


	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	boundary->loadCube(Vec3f(-1.5, 0, -1.5), Vec3f(1.5, 3, 1.5), 0.005f, true);
	boundary->loadSDF(getAssetPath() + "cloth_shell/table/table.sdf", false);

	auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>(0.3,8000,0.003,7e-4));
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
	scn->printModuleInfo(true);

	return scn;
}

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}


int main()
{
	Log::setUserReceiver(&RecieveLogMessage);

	GlfwApp window;
	window.setSceneGraph(createScene());

	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}