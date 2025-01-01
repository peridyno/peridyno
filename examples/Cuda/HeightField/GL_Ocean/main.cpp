#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/Ocean.h>
#include <HeightField/OceanPatch.h>
#include <HeightField/CapillaryWave.h>

#include <Mapping/HeightFieldToTriangleSet.h>

#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto oceanPatch = scn->addNode(std::make_shared<OceanPatch<DataType3f>>());
	oceanPatch->varWindType()->setValue(8);

	auto root = scn->addNode(std::make_shared<Ocean<DataType3f>>());
	root->varExtentX()->setValue(2);
	root->varExtentZ()->setValue(2);
	oceanPatch->connect(root->importOceanPatch());

	return scn;
}

int main()
{
	GlfwApp app;
	app.initialize(1024, 768);

	app.setSceneGraph(createScene());
	app.renderWindow()->getCamera()->setUnitScale(52);

	app.mainLoop();

	return 0;
}