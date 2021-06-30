#include "gtest/gtest.h"
#include "Framework/Node.h"
#include "Framework/Module.h"
using namespace dyno;

TEST(Object, id)
{
	Node n0;
	Node n1;

	Module m0;
	EXPECT_EQ(n0.objectId() < n1.objectId(), true);
	EXPECT_EQ(n1.objectId() < m0.objectId(), true);
}