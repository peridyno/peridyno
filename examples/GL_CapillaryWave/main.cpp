#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/CapillaryWave.h>

#include <Mapping/HeightFieldToTriangleSet.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<CapillaryWave<DataType3f>> root = scene.createNewScene<CapillaryWave<DataType3f>>(512,512.0f);

	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	root->currentTopology()->connect(mapper->inHeightField());
	root->graphicsPipeline()->pushModule(mapper);

	mapper->varScale()->setValue(0.1);
	mapper->varTranslation()->setValue(Vec3f(-2, 0.2, -2));

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