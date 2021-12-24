#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/Ocean.h>
#include <HeightField/OceanPatch.h>

#include "Mapping/HeightFieldToTriangleSet.h"

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

 	std::shared_ptr<OceanPatch<DataType3f>> root = scene.createNewScene<OceanPatch<DataType3f>>(512, 512.0f, 8);

	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	root->currentTopology()->connect(mapper->inHeightField());
	root->graphicsPipeline()->pushModule(mapper);

	mapper->varScale()->setValue(0.01);
	mapper->varTranslation()->setValue(Vec3f(0, 0.2, 0));

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(0, 0.2, 1.0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	root->graphicsPipeline()->pushModule(sRender);
}

int main()
{
	CreateScene();

	RenderEngine* engine = new GLRenderEngine;
	
	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1024, 768);
	window.mainLoop();

	delete engine;

	return 0;
}