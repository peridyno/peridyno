#include "gtest/gtest.h"
#include "Interval.h"

using namespace dyno;

TEST(Interval, function)
{
	Interval<Real> inter1(0, 1);
	Interval<Real> inter2(1, 2);

	Interval<Real> inter3 = inter1.intersect(inter2);

	EXPECT_EQ(inter3.leftLimit(), Real(1));
	EXPECT_EQ(inter3.rightLimit(), Real(1));
	EXPECT_EQ(inter3.isEmpty(), false);

	Interval<Real> inter4(0, 1, true, true);
	Interval<Real> inter5 = inter4.intersect(inter2);
	EXPECT_EQ(inter5.isEmpty(), true);

	Interval<Real> inter6(-1, 0);
	EXPECT_EQ(inter6.intersect(inter4).isEmpty(), true);
}
