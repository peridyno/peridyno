#include <QtApp.h>
using namespace dyno;

#include "Node.h"
#include "Vector.h"

/**
 * @brief This example demonstrates the resetting order after the field values in the node is updated.
 */

class Source : public Node
{
	DECLARE_CLASS(Source);
public:
	Source() {
	};
	~Source() {};

	DEF_VAR(int, Value, 1, "Define a scalar");

	DEF_VAR_OUT(int, Value, "Output value");

protected:
	void resetStates() {
		this->outValue()->setValue(this->varValue()->getData());

		std::cout << "Node " << this->getName() << " is reset, and the new value is " << this->varValue()->getData() << std::endl;
	}
};

IMPLEMENT_CLASS(Source);

class Summation : public Node
{
	DECLARE_CLASS(Summation);
public:
	Summation() {
	};
	~Summation() {};

	DEF_VAR(int, Value, 1, "Define a scalar");

	DEF_VAR_IN(int, Value, "Input value");

	DEF_VAR_OUT(int, Value, "Output value");

protected:
	void resetStates() {
		this->outValue()->setValue(this->varValue()->getData() + this->inValue()->getData());

		std::cout << "Node " << this->getName() << " is reset, and the new value is " << this->outValue()->getData() << std::endl;
	}
};

IMPLEMENT_CLASS(Summation);

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto source1 = scn->addNode(std::make_shared<Source>());
	source1->setName("Source1");

	auto sum1 = scn->addNode(std::make_shared<Summation>());
	sum1->setName("Sum1");

	auto sum2 = scn->addNode(std::make_shared<Summation>());
	sum2->setName("Sum2");

	source1->outValue()->connect(sum1->inValue());
	sum1->outValue()->connect(sum2->inValue());

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}