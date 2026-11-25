#include "gtest/gtest.h"
#include "Fraction.h"
#include "iostream"

using namespace dyno;

TEST(Fraction, arithmetic)
{
    Fraction<int> f0(2, 3);
    Fraction<int> f1(1, 6);

    auto sum = f0 + f1;

    EXPECT_EQ(sum == Fraction<int>(5, 6), true);

    auto sub = f0 - f1;
    EXPECT_EQ(sub == Fraction<int>(1, 2), true);

	auto mul = f0 * f1;
	EXPECT_EQ(mul == Fraction<int>(1, 9), true);

	auto div = f0 / f1;
	EXPECT_EQ(div == Fraction<int>(4, 1), true);
}

TEST(Fraction, compare)
{
    EXPECT_EQ(Fraction<int>(3, 4) > Fraction<int>(2, 3), true);
    EXPECT_EQ(Fraction<int>(3, 4) < Fraction<int>(2, 3), false);

    EXPECT_EQ(Fraction<int>(-3, 4) > Fraction<int>(-2, 3), false);
    EXPECT_EQ(Fraction<int>(-3, 4) < Fraction<int>(-2, 3), true);

    EXPECT_EQ(Fraction<int>(11, 12) <= Fraction<int>(8, 9), false);
    EXPECT_EQ(Fraction<int>(11, 12) >= Fraction<int>(8, 9), true);

    EXPECT_EQ(Fraction<int>(-3, 4) < 0, true);
    EXPECT_EQ(Fraction<int>(3, 4) > 0, true);

    Fraction<int> zero(0, 10);
    EXPECT_EQ(zero.numerator() == 0, true);
    EXPECT_EQ(zero.denominator() == 1, true);

    Fraction<int> zero1(0, -11);
    EXPECT_EQ(zero1.numerator() == 0, true);
    EXPECT_EQ(zero1.denominator() == 1, true);
}