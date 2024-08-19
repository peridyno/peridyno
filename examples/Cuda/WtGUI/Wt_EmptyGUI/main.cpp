#include "WtApp.h"

#include "SceneGraph.h"
#include "SphereModel.h"

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto sphere = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	sphere->varLocation()->setValue(Vec3f(0.6, 0.85, 0.5));
	sphere->varRadius()->setValue(0.1f);
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
