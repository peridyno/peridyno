#include <QtApp.h>
#include <GLRenderEngine.h>

#include "Node.h"
#include "Tuple.h"

using namespace dyno;

class MyTuple : public Tuple
{
public:
	MyTuple() {};

	DEF_VAR(bool, Boolean, false, "Define a boolean field");

	DEF_VAR(int, Int, 1, "Define an int");

	DEF_VAR(float, Float, 1.0f, "Define a float field");

	DEF_VAR(Vec3f, Vector, Vec3f(1.0f), "Define a vector field");
};

class MyNode : public Node
{
public:
	MyNode() {};
	~MyNode() override {};

	DEF_VAR(bool, AnotherBoolean, false, "Define a boolean field");

	DEF_TUPLE(MyTuple, Tuple, "Define a Tuple");
};

using namespace dyno;

int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a custom node
	scn->addNode(std::make_shared<MyNode>());


	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}
