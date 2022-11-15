#include <QtApp.h>
using namespace dyno;

#include "Node.h"
#include "Vector.h"

/**
 * @brief This example demonstrates how to show Chinese for both the node and fields
 */

class ChineseNode : public Node
{
public:
	ChineseNode() {
		this->varScalar()->setObjectName("标量");
		this->varVector()->setObjectName("矢量");

		this->stateTimeStep()->setObjectName("时间步长");
		this->stateElapsedTime()->setObjectName("时刻");
		this->stateFrameNumber()->setObjectName("当前帧");
	};
	~ChineseNode() {};

	std::string caption() override {
		return "测试中文";
	}

	DEF_VAR(float, Scalar, 1.0f, "Define a scalar");

	DEF_VAR(Vec3f, Vector, 0.0f, "Define a vector");
};

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto nickname = scn->addNode(std::make_shared<ChineseNode>());

	QtApp window;
	window.setSceneGraph(scn);
	window.createWindow(1366, 800);
	window.mainLoop();

	return 0;
}