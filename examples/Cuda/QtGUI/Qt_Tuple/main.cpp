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
	MyTuple() {
		this->varVec3fTupleArray()->pushBack(Vec3f());
	};

	//Deep copy
	MyTuple& operator=(MyTuple& other) {
		this->varBoolean()->setValue(other.varBoolean()->getValue());
		this->varInt()->setValue(other.varInt()->getValue());
		this->varFloat()->setValue(other.varFloat()->getValue());
		this->varVector()->setValue(other.varVector()->getValue());

		this->varVec3fTupleArray()->assign(other.varVec3fTupleArray());

		return *this;
	}

	DEF_VAR(bool, Boolean, false, "Define a boolean; QtStyle(HLayout,OnlyDetail)");

	DEF_VAR(int, Int, 1, "Define an int; QtStyle(HLayout,OnlyDetail)");

	DEF_VAR(float, Float, 1.0f, "Define a float field; QtStyle(HLayout,OnlyDetail)");

	DEF_VAR(Vec3f, Vector, Vec3f(1.0f), "Define a vector field");

	DEF_LIST(Vec3f, Vec3fTupleArray, "");
};

class MyNode : public Node
{
public:
	MyNode() 
	{
  		this->varFloatList()->pushBack(1.0f);
		this->varFloatList()->pushBack(3.1415926f);
		this->varFloatList()->pushBack(52.31f);

		this->varIntList()->pushBack(5);
		this->varIntList()->pushBack(10);
		this->varIntList()->pushBack(3);

		this->varTupleList()->pushBack(MyTuple());
		this->varTupleList()->pushBack(MyTuple());

		this->varTransformList()->pushBack(Transform3f());
		this->varTransformList()->pushBack(Transform3f());
		this->varTransformList()->pushBack(Transform3f());
	};

	~MyNode() override {};

	void resetStates() override
	{
		auto begin_it = this->varFloatList()->begin();
		for (auto it = begin_it; it != this->varFloatList()->end(); it++)
		{
			std::cout << this->varFloatList()->getElement(it) << std::endl;
		}

		std::cout << "Reset called " << std::endl;
	}

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

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}
