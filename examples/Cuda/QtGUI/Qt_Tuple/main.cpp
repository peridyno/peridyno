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

	//DEF_ARRAY_VAR(Vec3f, Vec3fTupleArray, "");

};

class MyNode : public Node
{
public:
	MyNode() 
	{
		this->varVec3fArrays()->assign(std::vector<Vec3f>{Vec3f(5),Vec3f(1), Vec3f(10), Vec3f(3)});
		this->varFloatArrays()->assign(std::vector<float>{1.0f, 3.1415926f,52.31f});
		this->varIntArrays()->assign(std::vector<int>{1,3,5,10});
	};
	~MyNode() override {};

	DEF_VAR(Vec3f, AnotherVec3f, Vec3f(), "Define a boolean field");

	DEF_TUPLE(MyTuple, Tuple, "Define a Tuple");

	DEF_ARRAY_VAR(Vec3f, Vec3fArray, "");

	DEF_VAR(bool, AnotherBoolean, false, "Define a boolean field");
	DEF_VAR(bool, BBoolean, false, "Define a boolean field");
	DEF_VAR(bool, CBoolean, false, "Define a boolean field");
	DEF_ARRAY_VAR(float, FloatArray,"Define a boolean field");
	DEF_ARRAY_VAR(int, IntArray,"Define a boolean field");


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
