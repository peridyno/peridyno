#include <GlfwApp.h>

#include <SceneGraph.h>

#include <Topology/TriangleSet.h>

#include <GLPointVisualModule.h>
#include <GLWireframeVisualModule.h>
#include <GLSurfaceVisualModule.h>


using namespace std;
using namespace dyno;

/**
 * @brief This example demonstrates how to define a triangle set for the SurfaceMesh node and use rendering modules for visualization
 * 
 */

class SurfaceMesh : public Node
{
public:
	SurfaceMesh() {

		// geometry
		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
		triSet->loadObjFile(getAssetPath() + "standard/standard_sphere.obj");
		triSet->update();
		this->stateTriangles()->setDataPtr(triSet);

		//Point visualizer
		auto pointRender = std::make_shared<GLPointVisualModule>();
		pointRender->varBaseColor()->setValue(Vec3f(1.0f, 0.0f, 0.0));
		pointRender->varPointSize()->setValue(0.02f);
		this->stateTriangles()->connect(pointRender->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender);

		//Wireframe visualizer
		auto edgeRender = std::make_shared<GLWireframeVisualModule>();
		edgeRender->varBaseColor()->setValue(Vec3f(0, 1, 0));
		edgeRender->varRenderMode()->getDataPtr()->setCurrentKey(GLWireframeVisualModule::LINE);
		edgeRender->varLineWidth()->setValue(3.f);
		this->stateTriangles()->connect(edgeRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(edgeRender);

		//Triangle visualizer
		auto triRender = std::make_shared<GLSurfaceVisualModule>();
		triRender->varBaseColor()->setValue(Vec3f(0, 0, 1));
		this->stateTriangles()->connect(triRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(triRender);
	};

public:
	DEF_INSTANCE_STATE(TriangleSet<DataType3f>, Triangles, "Topology");
};


std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto mesh = scn->addNode(std::make_shared<SurfaceMesh>());

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


