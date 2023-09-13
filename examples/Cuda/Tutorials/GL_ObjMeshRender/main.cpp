#include <GlfwApp.h>
#include <SceneGraph.h>

#include "ObjMesh.h"
#include <GLPhotorealisticRender.h>

using namespace std;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	if(false) {
		auto mesh = scn->addNode(std::make_shared<ObjMeshNode>());
		mesh->varFileName()->setValue(getAssetPath() + "obj/standard/cube.obj");
		mesh->varLocation()->setValue(Vec3f(-0.5f, 0.0f, 0.0f));
		mesh->varScale()->setValue(Vec3f(0.3f));

		auto render = mesh->graphicsPipeline()->createModule<GLPhotorealisticRender>();

		mesh->stateVertex()->connect(render->inVertex());
		mesh->stateNormal()->connect(render->inNormal());
		mesh->stateTexCoord()->connect(render->inTexCoord());

		mesh->stateShapes()->connect(render->inShapes());
		mesh->stateMaterials()->connect(render->inMaterials());
		mesh->graphicsPipeline()->pushModule(render);
	}

	if(true) {
		auto mesh = scn->addNode(std::make_shared<ObjMeshNode>());
		//mesh->varFileName()->setValue(getAssetPath() + "obj/standard/sphere.obj");
		mesh->varFileName()->setValue(getAssetPath() + "obj/eyeball/eyeball.obj");
		mesh->varLocation()->setValue(Vec3f(0.5f, 0.0f, 0.0f));
		mesh->varScale()->setValue(Vec3f(0.3f));

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
	GlfwApp app;
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


