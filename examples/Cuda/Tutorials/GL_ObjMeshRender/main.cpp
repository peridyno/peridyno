#include <GlfwApp.h>
#include <SceneGraph.h>

#include "ObjMesh.h"
#include <GLSurfaceVisualModule.h>

using namespace std;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	{
		auto mesh = scn->addNode(std::make_shared<ObjMeshNode>());
		mesh->load("data/cube.obj");
		auto triSet = mesh->outTriangleSet()->getDataPtr();
		triSet->scale(0.3f);
		triSet->translate({ -0.5, 0, 0.5 });
		mesh->update();

		auto render = std::make_shared<GLSurfaceVisualModule>();
		mesh->outTriangleSet()->connect(render->inTriangleSet());

		render->varUseVertexNormal()->setValue(true);
		mesh->outNormal()->connect(render->inNormal());
		mesh->outNormalIndex()->connect(render->inNormalIndex());

		render->varColorMode()->getDataPtr()->setCurrentKey(GLSurfaceVisualModule::CM_Texture);
		mesh->outTexCoord()->connect(render->inTexCoord());
		mesh->outTexCoordIndex()->connect(render->inTexCoordIndex());
		mesh->outColorTexture()->connect(render->inColorTexture());

		mesh->graphicsPipeline()->pushModule(render);
	}

	{
		auto mesh = scn->addNode(std::make_shared<ObjMeshNode>());
		mesh->load("data/sphere.obj");
		auto triSet = mesh->outTriangleSet()->getDataPtr();
		triSet->scale(0.3f);
		triSet->translate({ 0.2, 0.2, -0.2 });
		mesh->update();

		auto render = std::make_shared<GLSurfaceVisualModule>();
		mesh->outTriangleSet()->connect(render->inTriangleSet());

		render->varUseVertexNormal()->setValue(true);
		mesh->outNormal()->connect(render->inNormal());
		mesh->outNormalIndex()->connect(render->inNormalIndex());

		render->varColorMode()->getDataPtr()->setCurrentKey(GLSurfaceVisualModule::CM_Texture);
		mesh->outTexCoord()->connect(render->inTexCoord());
		mesh->outTexCoordIndex()->connect(render->inTexCoordIndex());
		mesh->outColorTexture()->connect(render->inColorTexture());

		mesh->graphicsPipeline()->pushModule(render);
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


