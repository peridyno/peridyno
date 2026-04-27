#include <QtApp.h>
#include <GLRenderEngine.h>

#include "Node.h"
#include "Field/FList.h"
#include "Tuple.h"
#include "Matrix/Matrix3x3.h"
#include "Matrix/Transform2x2.h"
#include "initializeModeling.h"


using namespace dyno;

class MyTuple : public Tuple
{
public:
	MyTuple() {};

	DEF_VAR(bool, Boolean, false, "Define a boolean field");

	DEF_VAR(int, Int, 1, "Define an int");

	DEF_VAR(float, Float, 1.0f, "Define a float field");

	DEF_VAR(Vec3f, Vector, Vec3f(1.0f), "Define a vector field");

	DEF_LIST(Vec3f, Vec3fTupleArray, "");
};

class MyNode : public Node
{
public:
	MyNode() 
	{
  		this->varFloatList()->insert(1.0f);
		this->varFloatList()->insert(3.1415926f);
		this->varFloatList()->insert(52.31f);

		this->varIntList()->insert(5);
		this->varIntList()->insert(10);
		this->varIntList()->insert(3);

		this->varTupleList()->insert(MyTuple());
		this->varTupleList()->insert(MyTuple());

		this->varTransformList()->insert(Transform3f());
		this->varTransformList()->insert(Transform3f());
		this->varTransformList()->insert(Transform3f());
	};

	~MyNode() override {};

	DEF_VAR(double, VarDouble,30.0f, "Define a list");
	DEF_VAR(float, VarFloat,30.0f, "Define a list");
	DEF_VAR(int, VarInt,10, "Define a list");
	DEF_VAR(uint, VarUint,10, "Define a list");

	DEF_TUPLE(MyTuple, Tuple, "Define a Tuple");
	DEF_LIST(float, FloatList, "Define a list");
	DEF_LIST(int, IntList, "Define a list");
	DEF_LIST(Transform3f, TransformList, "Define a list");
	DEF_LIST(MyTuple, TupleList, "");

};

using namespace dyno;

int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a custom node
	auto tupleNode = scn->addNode(std::make_shared<MyNode>());

// 	auto f = dynamic_cast<FCArray<Tuple>*> (tupleNode->varMyTuples());
// 	auto fm = dynamic_cast<FCArray<MyTuple>*> (tupleNode->varMyTuples());
// 	auto f2 = TypeInfo::cast<FCArray<Tuple>> (tupleNode->varMyTuples());
// 	auto fm2 = TypeInfo::cast<FCArray<MyTuple>> (tupleNode->varMyTuples());


	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}
