#include <QtApp.h>

#include <SceneGraph.h>

#include <HeightField/OceanPatch.h>
#include <HeightField/LargeOcean.h>
#include <HeightField/CapillaryWave.h>

#include <Mapping/HeightFieldToTriangleSet.h>

#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto patch = scn->addNode(std::make_shared<OceanPatch<DataType3f>>());
	patch->varWindType()->setValue(8);

	auto ocean = scn->addNode(std::make_shared<LargeOcean<DataType3f>>());
	patch->connect(ocean->importOceanPatch());

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);

	app.renderWindow()->getCamera()->setUnitScale(52);

	app.mainLoop();

	return 0;
}