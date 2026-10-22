#include <QtApp.h>
#include <GLRenderEngine.h>

#include "Node.h"
#include "Field/FList.h"
#include "Tuple.h"
#include "Matrix/Matrix3x3.h"
#include "Matrix/Transform2x2.h"
#include "initializeModeling.h"

#include "RigidBody/MultiBodyTuple.h"


using namespace dyno;


class MyNode : public Node
{
public:
	MyNode() 
	{
		auto callback = std::make_shared<FCallBackFunc>(std::bind(&MyNode::varChanged, this));

		this->varType()->attach(callback);
		this->varType()->setCurrentKey(1);

	};

	~MyNode() override {};

	void resetStates() override
	{

	}

	void varChanged() 
	{
		if (this->varType()->getDataPtr()->currentKey() == Type::A)
		{
			this->varVarDouble()->setValue(1.0,false);
			this->varVarFloat()->setValue(0.5,false);
			this->varVarInt()->setValue(0,false);
			this->varVarUint()->setValue(0,false);
		}
		else if(this->varType()->getDataPtr()->currentKey() == Type::B)
		{
			this->varVarDouble()->setValue(0.0, false);
			this->varVarFloat()->setValue(1.0, false);
			this->varVarInt()->setValue(1, false);
			this->varVarUint()->setValue(1, false);
		}


		std::cout << "Type: " << this->varType()->getValue().currentKey() <<";  D: " << this->varVarDouble()->getValue() << ";  F: " << this->varVarFloat()->getValue() << ";  I: " << this->varVarInt()->getValue() << ";  U: " << this->varVarUint()->getValue() << "\n";
	}

	DECLARE_ENUM(Type,
		A = 0,
		B = 1);

	DEF_ENUM(Type, Type, Type::A, "");

	DEF_VAR(double, VarDouble,30.0f, "Define a list");
	DEF_VAR(float, VarFloat,30.0f, "Define a list");
	DEF_VAR(int, VarInt,10, "Define a list");
	DEF_VAR(uint, VarUint,10, "Define a list");

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
