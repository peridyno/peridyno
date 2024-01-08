#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/LandScape.h>

#include "Mapping/HeightFieldToTriangleSet.h"


#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto land = scn->addNode(std::make_shared<LandScape<DataType3f>>());
	land->varFileName()->setValue(getAssetPath() + "landscape/Landscape_1 Map_1024x1024.png");
	land->varLocation()->setValue(Vec3f(0.0f, 100.0f, 0.0f));
	land->varScale()->setValue(Vec3f(1.0f, 64.0f, 1.0f));

	return scn;
}

int main()
{
	GlfwApp app;
	app.initialize(1024, 768);

	app.setSceneGraph(createScene());
	app.renderWindow()->getCamera()->setUnitScale(512);

	app.mainLoop();

	return 0;
}