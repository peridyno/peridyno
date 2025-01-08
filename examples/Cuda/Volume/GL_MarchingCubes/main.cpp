#include <UbiApp.h>

#include <SceneGraph.h>

#include <Volume/VolumeLoader.h>
#include <Volume/VolumeClipper.h>

#include "Volume/MarchingCubes.h"

#include "Node/GLSurfaceVisualNode.h"

#include "GLSurfaceVisualModule.h"

#include "ColorMapping.h"

#include <initializeIO.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createClipper()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto loader = scn->addNode(std::make_shared<VolumeLoader<DataType3f>>());
	loader->varFileName()->setValue(getAssetPath() + "bowl/bowl.sdf");

	auto clipper = scn->addNode(std::make_shared<VolumeClipper<DataType3f>>());
	loader->stateLevelSet()->connect(clipper->inLevelSet());

	return scn;
}

int main()
{
	dynoIO::initStaticPlugin();

	UbiApp app(GUIType::GUI_QT);

	app.setSceneGraph(createClipper());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}