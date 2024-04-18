#include <GlfwApp.h>

#include "GltfLoader.h"

#include "Module/GLPhotorealisticInstanceRender.h"

using namespace dyno;

class GenerateInstances : public Node
{
public:
	GenerateInstances() {
		this->stateTransform()->allocate();
		
	};

	void resetStates() override 
	{
		auto mesh = this->inTextureMesh()->constDataPtr();

		int copyNum = 0;
		int shapeNum = mesh->shapes().size();

		auto offest = this->varOffest()->getValue();

		std::vector<std::vector<Transform3f>> transform;
		transform.resize(shapeNum);
		for (size_t i = 0; i < shapeNum; i++)
		{
			for (size_t j = 0; j < copyNum; j++)
			{
				transform[i].push_back(Transform3f(Vec3f(offest[0]*j, 0, 0), Mat3f::identityMatrix(), Vec3f(1, 1, 1)));
				std::cout << transform[i][transform[i].size() - 1].translation().x << ", " << transform[i][transform[i].size() - 1].translation().y << ", " << transform[i][transform[i].size() - 1].translation().z << std::endl;
					
			}
			copyNum++;

		}

		auto tl = this->stateTransform()->getDataPtr();
		tl->assign(transform);
	}

	DEF_VAR(Vec3f, Offest,Vec3f(0.4,0,0) ,"");

	DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");

	DEF_ARRAYLIST_STATE(Transform3f, Transform, DeviceType::GPU, "");
};

int main()
{
	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "gltf/FlightHelmet/FlightHelmet.gltf");
	
	auto module = gltf->graphicsPipeline()->findFirstModule<GLWireframeVisualModule>();
	//gltf->deleteModule(module);
	gltf->graphicsPipeline()->clear();
	gltf->graphicsPipeline()->pushModule(module);

	auto instanceData = scn->addNode(std::make_shared<GenerateInstances>());
	gltf->stateTextureMesh()->connect(instanceData->inTextureMesh());

	auto render = std::make_shared<GLPhotorealisticInstanceRender>();
	gltf->stateTextureMesh()->connect(render->inTextureMesh());
	instanceData->stateTransform()->connect(render->inTransform());
	instanceData->graphicsPipeline()->pushModule(render);

	GlfwApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}