#include "gtest/gtest.h"
#include "ModuleFields.h"

using namespace dyno;

TEST(Field, connect)
{
	ModuleFields m1("m1");
	ModuleFields m2("m2");

	m1.varArea()->connect(m2.varArea());

	m1.varWidth()->setValue(12.0f);
	m1.varHeight()->setValue(5.0f);

	EXPECT_EQ(m1.varArea()->getValue(), float(60));
}
