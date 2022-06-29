#include <GlfwApp.h>
#include <GLRenderEngine.h>

#include "Array/Array.h"
#include "Matrix.h"
#include "Node.h"

#include "SceneGraph.h"
#include "GLSurfaceVisualModule.h"
#include "GLInstanceVisualModule.h"

#include "SurfacePickerNode.h"


using namespace dyno;

class Instances : public Node
{
public:
	Instances() {

		Transform3f tm;
		CArray<Transform3f> hTransform;
		for (uint i = 0; i < 5; i++)
		{
			tm.translation() = Vec3f(0.4 * i, 0, 0);
			tm.scale() = Vec3f(1.0 + 0.1 * i, 1.0 - 0.1 * i, 1.0);
			tm.rotation() = Quat<float>(i * (-0.2), Vec3f(1, 0, 0)).toMatrix3x3();
			hTransform.pushBack(tm);
		}

		this->stateTransforms()->allocate()->assign(hTransform);

		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
		triSet->loadObjFile("../../data/standard/standard_sphere.obj");

		this->stateTopology()->setDataPtr(triSet);

		hTransform.clear();
	};

		DEF_ARRAY_STATE(Transform3f, Transforms, DeviceType::GPU, "Instance transform");

};


int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto instanceNode = scn->addNode(std::make_shared<Instances>());

	auto surfacePickerNode = scn->addNode(std::make_shared<SurfacePickerNode<DataType3f>>());

	//Create a CustomMouseIteraction object to handle the mouse event,
	//Press/release the mouse button to show the information
	instanceNode->stateTopology()->connect(surfacePickerNode->inInTopology());



	auto instanceRender = std::make_shared<GLSurfaceVisualModule>();
	instanceRender->setColor(Vec3f(1, 0, 0));
	surfacePickerNode->stateSelectedTopology()->connect(instanceRender->inTriangleSet());
	surfacePickerNode->graphicsPipeline()->pushModule(instanceRender);

	auto instanceRender1 = std::make_shared<GLSurfaceVisualModule>();
	instanceRender1->setColor(Vec3f(0, 1, 1));
	surfacePickerNode->stateOtherTopology()->connect(instanceRender1->inTriangleSet());
	surfacePickerNode->graphicsPipeline()->pushModule(instanceRender1);

	scn->setUpperBound({ 4, 4, 4 });

	GlfwApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}
