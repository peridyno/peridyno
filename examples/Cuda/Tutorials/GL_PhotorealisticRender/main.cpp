#include <GlfwApp.h>
#include <SceneGraph.h>

#include <TextureMeshLoader.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	if(true) {
		auto mesh = scn->addNode(std::make_shared<TextureMeshLoader>());
		mesh->varFileName()->setValue(getAssetPath() + "obj/standard/cube.obj");
		mesh->varScale()->setValue(Vec3f(0.3f));
		mesh->varLocation()->setValue(Vec3f(-1.5f, 0.3f, 0.0f));
	}

	if(true) {
		auto mesh = scn->addNode(std::make_shared<TextureMeshLoader>());
		mesh->varFileName()->setValue(getAssetPath() + "obj/moon/Moon_Normal.obj");
		//mesh->varFileName()->setValue("C:/Users/M/Desktop/land/Landscape.obj");

		mesh->varScale()->setValue(Vec3f(0.005f));
		mesh->varLocation()->setValue(Vec3f(0.5f, 0.3f, 0.5f));
	}

	return scn;
}

int main()
{
#ifdef VK_BACKEND
	VkSystem::instance()->initialize();
#endif

	GlfwApp app;
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


