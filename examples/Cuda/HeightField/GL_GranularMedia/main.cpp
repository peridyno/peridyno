#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/GranularMedia.h>

#include "Mapping/HeightFieldToTriangleSet.h"

#include <GLSurfaceVisualModule.h>

#include <HeightField/SurfaceParticleTracking.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto root = scn->addNode(std::make_shared<GranularMedia<DataType3f>>());

	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	root->stateHeightField()->connect(mapper->inHeightField());
	root->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(0.8, 0.8, 0.8));
	sRender->varUseVertexNormal()->setValue(true);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	root->graphicsPipeline()->pushModule(sRender);

	auto tracking = scn->addNode(std::make_shared<SurfaceParticleTracking<DataType3f>>());
	root->connect(tracking->importGranularMedia());

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