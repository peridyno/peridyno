#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/Ocean.h>
#include <HeightField/OceanPatch.h>
#include <HeightField/CapillaryWave.h>

#include "Mapping/HeightFieldToTriangleSet.h"


#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto root = scn->addNode(std::make_shared<Ocean<DataType3f>>());
	
	auto oceanPatch = scn->addNode(std::make_shared<OceanPatch<DataType3f>>(512, 512, 4));
	oceanPatch->connect(root->importOceanPatch());

	auto capillaryWave = scn->addNode(std::make_shared<CapillaryWave<DataType3f>>(512, 512.0f));
	capillaryWave->connect(root->importCapillaryWaves());
	

	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	mapper->varScale()->setValue(0.01);
	mapper->varTranslation()->setValue(Vec3f(0, 0.2, 0));

	root->stateTopology()->connect(mapper->inHeightField());
	root->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(0, 0.2, 1.0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	root->graphicsPipeline()->pushModule(sRender);

	return scn;
}

int main()
{
	GlfwApp window;
	window.setSceneGraph(createScene());
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}