#include <QtApp.h>
#include <SceneGraph.h>

#include "ObjMesh.h"
#include <GLSurfaceVisualModule.h>
#include <GLPhotorealisticRender.h>

using namespace std;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	{
		auto mesh = scn->addNode(std::make_shared<ObjMeshNode>());
		mesh->varFileName()->setValue(getAssetPath() + "obj/standard/cube.obj");
		mesh->varLocation()->setValue(Vec3f(-0.5f, 0.0f, 0.0f));
		mesh->varScale()->setValue(Vec3f(0.3f));

		auto render = std::make_shared<GLSurfaceVisualModule>();
		mesh->stateTriangleSet()->connect(render->inTriangleSet());

		render->varUseVertexNormal()->setValue(true);
		mesh->stateNormal()->connect(render->inNormal());
		mesh->stateNormalIndex()->connect(render->inNormalIndex());

		render->varColorMode()->setCurrentKey(GLSurfaceVisualModule::CM_Texture);
		mesh->stateTexCoord()->connect(render->inTexCoord());
		mesh->stateTexCoordIndex()->connect(render->inTexCoordIndex());
		mesh->stateColorTexture()->connect(render->inColorTexture());

		mesh->graphicsPipeline()->pushModule(render);
	}

	{
		auto mesh = scn->addNode(std::make_shared<ObjMeshNode>());
		//mesh->varFileName()->setValue(getAssetPath() + "obj/standard/sphere.obj");
		mesh->varFileName()->setValue(getAssetPath() + "obj/eyeball/eyeball.obj");
		mesh->varLocation()->setValue(Vec3f(0.5f, 0.0f, 0.0f));
		mesh->varScale()->setValue(Vec3f(0.3f));

		auto render = std::make_shared<GLSurfaceVisualModule>();
		mesh->stateTriangleSet()->connect(render->inTriangleSet());

		render->varUseVertexNormal()->setValue(true);
		mesh->stateNormal()->connect(render->inNormal());
		mesh->stateNormalIndex()->connect(render->inNormalIndex());

		render->varColorMode()->setCurrentKey(GLSurfaceVisualModule::CM_Texture);
		mesh->stateTexCoord()->connect(render->inTexCoord());
		mesh->stateTexCoordIndex()->connect(render->inTexCoordIndex());
		mesh->stateColorTexture()->connect(render->inColorTexture());

		mesh->graphicsPipeline()->pushModule(render);

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
	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


