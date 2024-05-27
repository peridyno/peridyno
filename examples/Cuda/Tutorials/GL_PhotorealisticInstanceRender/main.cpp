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
		const int instanceCount = 8;
		const int shapeNum = mesh->shapes().size();

		std::vector<std::vector<Transform3f>> transform(shapeNum);

		for (size_t j = 0; j < instanceCount; j++)
		{
			auto rotate = Quat1f(j * 2.f * M_PI / instanceCount, Vec3f(0, 1, 0)).toMatrix3x3();
			auto scale = Vec3f(0.8 + float(j) / instanceCount);
			auto translate = rotate * Vec3f(0, 0, 1) + scale * Vec3f(0, 0.32, 0);

			for (size_t i = 0; i < shapeNum; i++) {

				auto shapeTransform = this->inTextureMesh()->constDataPtr()->shapes()[i]->boundingTransform;

				transform[i].push_back(Transform3f(translate + rotate * shapeTransform.translation() * scale, rotate, scale));
			}
		}

		auto tl = this->stateTransform()->getDataPtr();
		tl->assign(transform);
	}

	//DEF_VAR(Vec3f, Offest, Vec3f(0.4, 0, 0), "");

	DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");
	DEF_ARRAYLIST_STATE(Transform3f, Transform, DeviceType::GPU, "");
};

int main()
{
	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-3, 0, -3));
	scn->setUpperBound(Vec3f( 3, 3,  3));

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