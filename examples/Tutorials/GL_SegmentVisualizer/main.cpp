#include <GlfwApp.h>
#include <GLRenderEngine.h>

#include "SceneGraph.h"

#include "Topology/HexahedronSet.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

using namespace dyno;

class HexahedronNode : public Node
{
public:
	HexahedronNode() {

		// geometry
		std::shared_ptr<HexahedronSet<DataType3f>> hexSet = std::make_shared<HexahedronSet<DataType3f>>();
		this->stateHexahedrons()->setDataPtr(hexSet);

		int num = 50;
		float s = 0.01;

		int num2 = 2 * num + 1;

		std::vector<Vec3f> coords;
		for (int k = -num; k <= num; k++)
		{
			for (int j = -num; j <= num; j++)
			{
				for (int i = -num; i <= num; i++)
				{
					coords.push_back(Vec3f(i * s, j * s, k * s));
				}
			}
		}

		auto index = [=](int i, int j, int k) ->int {return i + j * num2 + k * num2 * num2; };

		std::vector<TopologyModule::Hexahedron> hex;
		for (int k = 0; k < num2; k++)
		{
			for (int j = 0; j < num2; j++)
			{
				for (int i = 0; i < num2; i++)
				{
					int v0 = index(i, j, k);
					int v1 = index(i, j, k + 1);
					int v2 = index(i + 1, j, k + 1);
					int v3 = index(i + 1, j, k);
					int v4 = index(i, j + 1, k);
					int v5 = index(i, j + 1, k + 1);
					int v6 = index(i + 1, j + 1, k + 1);
					int v7 = index(i + 1, j + 1, k);

					hex.push_back(TopologyModule::Hexahedron(v0, v1, v2, v3, v4, v5, v6, v7));
				}
			}
		}

		hexSet->setPoints(coords);
		hexSet->setHexahedrons(hex);

		hexSet->update();
	};

	DEF_INSTANCE_STATE(HexahedronSet<DataType3f>, Hexahedrons, "Topology");
};

int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	// create a hexahedron node
	auto instanceNode = scn->addNode(std::make_shared<HexahedronNode>());

	// wireframe rendering
	auto edgeRender = std::make_shared<GLWireframeVisualModule>();
	edgeRender->setColor(Vec3f(0, 1, 0));
	edgeRender->setEdgeMode(GLWireframeVisualModule::LINE);
	edgeRender->varLineWidth()->setValue(2.f);
	instanceNode->stateHexahedrons()->connect(edgeRender->inEdgeSet());
	instanceNode->graphicsPipeline()->pushModule(edgeRender);

// 	auto ptRender = std::make_shared<GLPointVisualModule>();
// 	ptRender->setColor(Vec3f(1, 0, 0));
// 	instanceNode->stateHexahedrons()->connect(ptRender->inPointSet());
// 	instanceNode->graphicsPipeline()->pushModule(ptRender);

	GlfwApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}
