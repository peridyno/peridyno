#include <GlfwApp.h>

#include <SceneGraph.h>
#include <GLWireframeVisualModule.h>

using namespace dyno;

class Spring : public Node
{
public:
	Spring() {
		auto triSet = std::make_shared<EdgeSet>();
		this->stateTopology()->setDataPtr(triSet);
	}
	virtual ~Spring() {};

	DEF_INSTANCE_STATE(EdgeSet, Topology, "");

protected:
	void resetStates() override {
		std::vector<Vec3f> vertices;
		vertices.push_back(Vec3f(0.0));
		vertices.push_back(Vec3f(0.0, 1.0, 0.0));

		std::vector<TopologyModule::Edge> edges;
		edges.push_back(TopologyModule::Edge(0, 1));

		auto topo = this->stateTopology()->getDataPtr();
		topo->mPoints.assign(vertices);
		topo->mEdgeIndex.assign(edges);

		vertices.clear();
		edges.clear();
	}
};

std::shared_ptr<SceneGraph> createScene()
{
	auto scene = std::make_shared<SceneGraph>();

	auto cloth = scene->addNode(std::make_shared<Spring>());

	auto clothRender = std::make_shared<GLWireframeVisualModule>();
	cloth->stateTopology()->connect(clothRender->inEdgeSet());
	cloth->graphicsPipeline()->pushModule(clothRender);

	return scene;
}

int main(int, char**)
{
	VkSystem::instance()->initialize();

	GlfwApp window;
	window.initialize(1024, 768);
	window.setSceneGraph(createScene());
	window.mainLoop();
	return 0;
}
