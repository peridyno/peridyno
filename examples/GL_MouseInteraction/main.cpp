#include <GlfwApp.h>
#include <GLRenderEngine.h>

#include "Array/Array.h"
#include "Matrix.h"
#include "Node.h"

#include "SceneGraph.h"
#include "GLSurfaceVisualModule.h"
#include "GLInstanceVisualModule.h"

#include "CustomMouseInteraction.h"

#include "Module/MouseIntersect.h"

using namespace dyno;

class Instances : public Node
{
public:
	Instances() {

		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
		triSet->loadObjFile("../../data/armadillo/armadillo.obj");

		this->stateTopology()->setDataPtr(triSet);

	};
};

int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto instanceNode = scn->addNode(std::make_shared<Instances>());

	//Create a CustomMouseIteraction object to handle the mouse event,
	//Press/release the mouse button to show the information
	auto mouseInterator = std::make_shared<CustomMouseIteraction>();
	instanceNode->stateTopology()->connect(mouseInterator->inTopology());
	instanceNode->animationPipeline()->pushModule(mouseInterator);

	auto mouseIntersecter = std::make_shared<MouseIntersect<DataType3f>>();
	mouseInterator->stateMouseRay()->connect(mouseIntersecter->inMouseRay());
	instanceNode->stateTopology()->connect(mouseIntersecter->inInitialTriangleSet());
	mouseInterator->stateMouseIntersect()->setDataPtr(mouseIntersecter);

	auto instanceRender = std::make_shared<GLSurfaceVisualModule>();
	instanceRender->setColor(Vec3f(0, 1, 0));
	mouseIntersecter->stateSelectedTriangleSet()->connect(instanceRender->inTriangleSet());
	mouseIntersecter->graphicsPipeline()->pushModule(instanceRender);

	auto instanceRender1 = std::make_shared<GLSurfaceVisualModule>();
	instanceRender1->setColor(Vec3f(0, 1, 1));
	mouseIntersecter->stateSelectedTriangleSet()->connect(instanceRender1->inTriangleSet());
	mouseIntersecter->graphicsPipeline()->pushModule(instanceRender1);

	scn->setUpperBound({ 4, 4, 4 });

	GlfwApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}
