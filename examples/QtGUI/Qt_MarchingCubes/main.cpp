#include <QtApp.h>

#include <SceneGraph.h>

#include "Mapping/MarchingCubes.h"

#include "Node/GLSurfaceVisualNode.h"

#include "Mapping/VolumeClipper.h"
#include "GLSurfaceVisualModule.h"

#include "ColorMapping.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto marchingCubes = scn->addNode(std::make_shared<MarchingCubes<DataType3f>>());

	auto isoSurfaceVisualizer = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	marchingCubes->outTriangleSet()->connect(isoSurfaceVisualizer->inTriangleSet());

	return scn;
}

std::shared_ptr<SceneGraph> createClipper()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto clipper = scn->addNode(std::make_shared<VolumeClipper<DataType3f>>());

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMin()->setValue(-0.5);
	colorMapper->varMax()->setValue(0.5);
	clipper->stateField()->connect(colorMapper->inScalar());
	clipper->graphicsPipeline()->pushModule(colorMapper);
// 
// 
 	auto surfaceVisualizer = std::make_shared<GLSurfaceVisualModule>();
	surfaceVisualizer->varColorMode()->getDataPtr()->setCurrentKey(1);
 	colorMapper->outColor()->connect(surfaceVisualizer->inColor());
	clipper->stateTriangleSet()->connect(surfaceVisualizer->inTriangleSet());
	clipper->graphicsPipeline()->pushModule(surfaceVisualizer);

	return scn;
}

int main()
{
	QtApp app;

	app.setSceneGraph(createClipper());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


