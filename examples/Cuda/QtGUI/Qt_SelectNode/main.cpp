#include <QtApp.h>
#include <GLRenderEngine.h>

#include <BasicShapes/SphereModel.h>

using namespace dyno;

int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a sphere
	auto sphere0 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	sphere0->varLocation()->setValue(Vec3f(-0.5f, 0.1f, 0.0f));
	sphere0->varRadius()->setValue(0.2f);

	//Create a sphere
	auto sphere1 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	sphere1->varLocation()->setValue(Vec3f(0.5f, 0.1f, 0.0f));
	sphere1->varRadius()->setValue(0.2f);

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}
