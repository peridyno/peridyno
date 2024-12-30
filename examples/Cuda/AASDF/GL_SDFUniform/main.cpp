#include <UbiApp.h>
#include <SceneGraph.h>

#include <Volume/VolumeGenerator.h>
#include <Volume/VolumeClipper.h>

#include <BasicShapes/SphereModel.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(2, 2, 2));
	scn->setLowerBound(Vec3f(-2, -2, -2));

	auto sphere = scn->addNode(std::make_shared<SphereModel<DataType3f>>());

	auto volume = scn->addNode(std::make_shared<VolumeGenerator<DataType3f>>());
	volume->varPadding()->setValue(10);
	volume->varSpacing()->setValue(0.05f);

	sphere->stateTriangleSet()->connect(volume->inTriangleSet());

	auto clipper = scn->addNode(std::make_shared<VolumeClipper<DataType3f>>()); ;
	volume->outLevelSet()->connect(clipper->inLevelSet());

	return scn;
}

int main()
{
	UbiApp window(GUIType::GUI_QT);

	window.setSceneGraph(createScene());
	// window.createWindow(2048, 1152);
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}


