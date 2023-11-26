#include <QtApp.h>
#include <SceneGraph.h>

#include "TexturedMesh.h"
#include <GLPhotorealisticRender.h>

using namespace std;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	if(true) {
		auto mesh = scn->addNode(std::make_shared<TexturedMesh>());
		mesh->varFileName()->setValue(getAssetPath() + "obj/standard/cube.obj");
		mesh->varScale()->setValue(Vec3f(0.3f));
		mesh->varLocation()->setValue(Vec3f(-1.5f, 0.3f, 0.0f));

		auto render = mesh->graphicsPipeline()->createModule<GLPhotorealisticRender>();

		mesh->stateVertex()->connect(render->inVertex());
		mesh->stateNormal()->connect(render->inNormal());
		mesh->stateTexCoord()->connect(render->inTexCoord());

		mesh->stateShapes()->connect(render->inShapes());
		mesh->stateMaterials()->connect(render->inMaterials());
		mesh->graphicsPipeline()->pushModule(render);
	}

	if(true) {
		auto mesh = scn->addNode(std::make_shared<TexturedMesh>());
		mesh->varFileName()->setValue(getAssetPath() + "obj/moon/Moon_Normal.obj");
		//mesh->varFileName()->setValue("C:/Users/M/Desktop/land/Landscape.obj");

		mesh->varScale()->setValue(Vec3f(0.005f));
		mesh->varLocation()->setValue(Vec3f(0.5f, 0.3f, 0.5f));

		auto realisticRender = mesh->graphicsPipeline()->createModule<GLPhotorealisticRender>();
		mesh->stateVertex()->connect(realisticRender->inVertex());
		mesh->stateNormal()->connect(realisticRender->inNormal());
		mesh->stateTexCoord()->connect(realisticRender->inTexCoord());
		mesh->stateShapes()->connect(realisticRender->inShapes());
		mesh->stateMaterials()->connect(realisticRender->inMaterials());
	}

	return scn;
}

int main()
{
#ifdef VK_BACKEND
	VkSystem::instance()->initialize();
#endif

	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


