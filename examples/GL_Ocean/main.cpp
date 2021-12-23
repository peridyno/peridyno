#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/OceanPatch.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

 	std::shared_ptr<OceanPatch> root = scene.createNewScene<OceanPatch>(512, 512.0f);
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