#include <GlfwApp.h>
#include <GLRenderEngine.h>

#include "Array/Array.h"
#include "Matrix.h"
#include "Node.h"

#include "SceneGraph.h"
#include "GLInstanceVisualModule.h"

using namespace dyno;

class Instances : public Node
{
public:
	Instances() {

		// geometry
		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
		triSet->loadObjFile(getAssetPath() + "armadillo/armadillo.obj");
		this->stateTopology()->setDataPtr(triSet);

		// instance transforms and colors
		CArray<Transform3f> instanceTransforms;
		CArray<Vec3f>		instanceColors;

		for (uint i = 0; i < 5; i++)
		{
			Transform3f tm;
			tm.translation()	= Vec3f(0.4 * i, 0, 0);
			tm.scale()			= Vec3f(1.0 + 0.1*i, 1.0 - 0.1*i, 1.0);
			tm.rotation()		= Quat<float>(i * (-0.2), Vec3f(1, 0, 0)).toMatrix3x3();
			instanceTransforms.pushBack(tm);

			instanceColors.pushBack(Vec3f(i * 0.2f, i * 0.2f, 1.f - i * 0.1f));
		}

		this->stateTransforms()->allocate()->assign(instanceTransforms);
		this->stateColors()->allocate()->assign(instanceColors);

		instanceTransforms.clear();
	};

	DEF_ARRAY_STATE(Transform3f, Transforms, DeviceType::GPU, "Instance transform");
	DEF_ARRAY_STATE(Vec3f, Colors, DeviceType::GPU, "Instance color");

	DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
};

int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	// create a instance node
	auto instanceNode = scn->addNode(std::make_shared<Instances>());

	// config instance rendering
	auto instanceRender = std::make_shared<GLInstanceVisualModule>();
	instanceRender->setColor(Vec3f(0, 1, 0));
	//instanceRender->setAlpha(0.5f);
	//instanceRender->varUseVertexNormal()->setValue(true);

	instanceNode->stateTopology()->connect(instanceRender->inTriangleSet());
	instanceNode->stateTransforms()->connect(instanceRender->inInstanceTransform());
	instanceNode->stateColors()->connect(instanceRender->inInstanceColor());
	instanceNode->graphicsPipeline()->pushModule(instanceRender);

	scn->setUpperBound({ 4, 4, 4});

	GlfwApp app;
	app.setSceneGraph(scn);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}
