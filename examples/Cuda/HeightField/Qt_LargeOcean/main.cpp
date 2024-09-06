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

	//Visualize the OceanPatch
	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	patch->stateHeightField()->connect(mapper->inHeightField());
	patch->graphicsPipeline()->pushModule(mapper);

	auto patchRender = std::make_shared<GLSurfaceVisualModule>();
	patchRender->setColor(Color(0, 0.2, 1.0));
	patchRender->varUseVertexNormal()->setValue(true);
	mapper->outTriangleSet()->connect(patchRender->inTriangleSet());
	patch->graphicsPipeline()->pushModule(patchRender);

	auto ocean = scn->addNode(std::make_shared<LargeOcean<DataType3f>>());
	patch->connect(ocean->importOceanPatch());

	auto oceanRender = std::make_shared<GLSurfaceVisualModule>();
	oceanRender->setColor(Color(0, 0.2, 1.0));
	oceanRender->varUseVertexNormal()->setValue(true);
	ocean->stateTriangleSet()->connect(oceanRender->inTriangleSet());
	ocean->stateTexCoord()->connect(oceanRender->inTexCoord());
	ocean->stateTexCoordIndex()->connect(oceanRender->inTexCoordIndex());
	ocean->stateBumpMap()->connect(oceanRender->inBumpMap());
	ocean->graphicsPipeline()->pushModule(oceanRender);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	ocean->stateTriangleSet()->connect(wireRender->inEdgeSet());
	ocean->graphicsPipeline()->pushModule(wireRender);

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