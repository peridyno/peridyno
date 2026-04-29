#include "gtest/gtest.h"
#include "Field/FList.h"
#include <vector>

using namespace dyno;

class MyTuple : public Tuple
{
public:
	MyTuple() {
	};

	//Deep copy
	MyTuple& operator=(MyTuple& other) {
		this->varBoolean()->setValue(other.varBoolean()->getValue());
		this->varInt()->setValue(other.varInt()->getValue());
		this->varFloat()->setValue(other.varFloat()->getValue());
		this->varVector()->setValue(other.varVector()->getValue());

		this->varIntList()->assign(other.varIntList());

		return *this;
	}

	DEF_VAR(bool, Boolean, false, "Define a boolean field");

	DEF_VAR(int, Int, 1, "Define an int");

	DEF_VAR(float, Float, 1.0f, "Define a float field");

	DEF_VAR(Vec3f, Vector, Vec3f(1.0f), "Define a vector field");

	DEF_LIST(int, IntList, "");
};

TEST(Tuple, assign)
{
	MyTuple tuple0;
	tuple0.varIntList()->pushBack(0);
	tuple0.varIntList()->pushBack(1);

	MyTuple tuple1;
	tuple1 = tuple0;

	EXPECT_EQ(tuple1.varIntList()->size() == 2, true);
}
