#include "gtest/gtest.h"
#include "VkProgram.h"

using namespace dyno;

TEST(VkProgram, iDivUp)
{
	uint dim1 = iDivUp(1, 5);
	EXPECT_EQ(dim1 == 1, true);

	uint dim2 = iDivUp(5, 5);
	EXPECT_EQ(dim2 == 1, true);
}