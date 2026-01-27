#include "UbiApp.h"
#include <SceneGraph.h>

#include <TextureMeshLoader.h>
#include "initializeModeling.h"
#include "BasicShapes/SphereModel.h"
#include "GltfLoader.h"

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	if(false) {
		auto mesh = scn->addNode(std::make_shared<TextureMeshLoader>());
		mesh->varFileName()->setValue(getAssetPath() + "obj/standard/cube.obj");
		mesh->varScale()->setValue(Vec3f(0.3f));
		mesh->varLocation()->setValue(Vec3f(-1.5f, 0.3f, 0.0f));
	}

	if(1) {
		auto mesh = scn->addNode(std::make_shared<TextureMeshLoader>());
		mesh->varFileName()->setValue(getAssetPath() + "obj/moon/Moon_Normal.obj");
		//mesh->varFileName()->setValue("C:/Users/M/Desktop/land/Landscape.obj");

		mesh->varScale()->setValue(Vec3f(0.005f));
		mesh->varLocation()->setValue(Vec3f(1.5f, 1.3f, 0.5f));
	}

	if (1) {
		auto mesh = scn->addNode(std::make_shared<TextureMeshLoader>());
		mesh->varFileName()->setValue(std::string("C:/Users/win11/Desktop/testMeetMat/MeetMat.obj"));
		//mesh->varFileName()->setValue("C:/Users/M/Desktop/land/Landscape.obj");

		mesh->varScale()->setValue(Vec3f(0.000f));
		mesh->varLocation()->setValue(Vec3f(0.0f, 1.0f, 0.0f));
	}

	//scn->addNode(std::make_shared<SphereModel<DataType3f>>());

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "gltf/FlightHelmet/FlightHelmet.gltf");
	gltf->varLocation()->setValue(Vec3f(0,0.5,0));

	return scn;
}

int main()
{
#ifdef VK_BACKEND
	VkSystem::instance()->initialize();
#endif
	Modeling::initStaticPlugin();
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


