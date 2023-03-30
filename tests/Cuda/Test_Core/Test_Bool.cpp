#include "gtest/gtest.h"
#include "Platform.h"
using namespace dyno;

TEST(Bool, func)
{
	Bool b1;
	EXPECT_EQ(b1 == false, true);

	Bool b2(false);
	Bool b3(true);

	b2 = false;
	b3 = true;

	EXPECT_EQ((b2&b3) == false, true);
	EXPECT_EQ((b2|b3) == true, true);

	EXPECT_EQ((b2&&b3) == false, true);
	EXPECT_EQ((b2||b3) == true, true);

	EXPECT_EQ(b2, false);
	EXPECT_EQ(b3, true);

	EXPECT_EQ(false || b2, false);
	EXPECT_EQ(true || b2, true);
}