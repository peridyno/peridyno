#include <QtApp.h>

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

int main()
{
	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1366, 800);
	app.setWindowTitle("Empty Qt-based GUI");
	app.mainLoop();

	return 0;
}