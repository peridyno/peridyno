#include "WtApp.h"

#include "SceneGraph.h"

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	return scn;
}

int main(int argc, char** argv)
{
	WtApp app;

	app.setSceneGraphCreator(&createScene);
	app.setSceneGraph(createScene());
	app.mainLoop();

	return 0;
}
