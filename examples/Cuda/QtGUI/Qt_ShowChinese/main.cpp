#include <QtApp.h>
using namespace dyno;

#include "Node.h"
#include "Vector.h"

/**
 * @brief This example demonstrates how to show Chinese for both the node and fields
 */

class ChineseNode : public Node
{
	DECLARE_CLASS(ChineseNode);
public:
	ChineseNode() {
		this->varScalar()->setObjectName(u8"标量");
		this->varVector()->setObjectName(u8"矢量");

		this->stateTimeStep()->setObjectName(u8"时间步长");
		this->stateElapsedTime()->setObjectName(u8"时刻");
		this->stateFrameNumber()->setObjectName(u8"当前帧");
	};
	~ChineseNode() {};

	std::string caption() override {
		return u8"测试中文";
	}

	std::string description() override {
		return u8"这是一个中文节点";
	}

	std::string getNodeType() override {
		return u8"中文节点";
	}

	DEF_VAR(float, Scalar, 1.0f, "Define a scalar");

	DEF_VAR(Vec3f, Vector, 0.0f, "Define a vector");
};

IMPLEMENT_CLASS(ChineseNode);

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto nickname = scn->addNode(std::make_shared<ChineseNode>());

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}