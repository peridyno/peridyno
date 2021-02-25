#include "gtest/gtest.h"
#include "Topology/TriangleSet.h"

using namespace dyno;

TEST(Key, construction)
{
	TriangleSet<DataType3f>::TKey tKey(2, 3, 1);

	EXPECT_EQ(tKey[0], 1);
	EXPECT_EQ(tKey[1], 2);
	EXPECT_EQ(tKey[2], 3);

	TriangleSet<DataType3f>::TKey tKey_1(3, 2, 1);

	EXPECT_EQ(tKey_1[0], 1);
	EXPECT_EQ(tKey_1[1], 2);
	EXPECT_EQ(tKey_1[2], 3);

	TriangleSet<DataType3f>::TKey tKey_2(293, 262, 294);

	EXPECT_EQ(tKey_2[0], 262);
	EXPECT_EQ(tKey_2[1], 293);
	EXPECT_EQ(tKey_2[2], 294);

	EXPECT_EQ(tKey < tKey_1, false);
	EXPECT_EQ(tKey > tKey_1, false);
	EXPECT_EQ(tKey == tKey_1, true);

	EdgeSet<DataType3f>::EKey eKey(3, 2);

	EXPECT_EQ(eKey[0], 2);
	EXPECT_EQ(eKey[1], 3);

	EdgeSet<DataType3f>::EKey eKey_1(1, 2);

	EXPECT_EQ(eKey_1[0], 1);
	EXPECT_EQ(eKey_1[1], 2);

	EXPECT_EQ(eKey < eKey_1, false);
	EXPECT_EQ(eKey > eKey_1, true);
}
