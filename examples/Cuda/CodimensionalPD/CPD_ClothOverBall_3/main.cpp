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
	object->varFileName()->setValue(getAssetPath() + "cloth_shell/ball/ball_model.obj");


	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	boundary->loadCube(Vec3f(-1.5, 0, -1.5), Vec3f(1.5, 3, 1.5), 0.005f, true);
	
	boundary->loadSDF(getAssetPath() + "cloth_shell/ball/ball_small_size_15.sdf", false);

	auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>(0.15,500,0.0005,1e-3));
	cloth->loadSurface(getAssetPath() + "cloth_shell/cloth_size_17_alt/cloth_40k_3.obj");
	cloth->connect(boundary->importTriangularSystems());
	cloth->setGrad_ite_eps(0);
	cloth->setMaxIteNumber(10);
	auto surfaceRendererCloth = std::make_shared<GLSurfaceVisualModule>();
	surfaceRendererCloth->setColor(Color(0.4, 0.4, 1.0));

	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
	surfaceRenderer->setColor(Color(0.4, 0.4, 0.4));
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