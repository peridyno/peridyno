#include <GlfwApp.h>
#include <SceneGraph.h>

#include "ObjMesh.h"
#include "GLObjMeshVisualModule.h"

using namespace std;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto render = std::make_shared<GLObjMeshVisualModule>();

	auto mesh = scn->addNode(std::make_shared<ObjMeshNode>());
	mesh->load("data/sphere.obj");
	//mesh->load("Landscape_Mesh_PolyReduce.obj");
	// geometry
	mesh->outPosition()->connect(render->inPosition());
	mesh->outNormal()->connect(render->inNormal());
	mesh->outTexCoord()->connect(render->inTexCoord());
	mesh->outIndex()->connect(render->inIndex());

	// texture
	mesh->outTexColor()->connect(render->inTexColor());
	//render->setColorTexture("Colormap_0.png");

	mesh->update();

	mesh->graphicsPipeline()->pushModule(render);

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


