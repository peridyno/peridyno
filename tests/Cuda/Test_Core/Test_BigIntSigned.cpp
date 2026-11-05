#include "gtest/gtest.h"
#include "Basic/BigIntSigned.h"

using namespace dyno;

// ============================================================
// int128_t tests
// ============================================================

TEST(Int128Signed, DefaultConstruction)
{
    int128_t a;
    EXPECT_EQ(a.lo, 0ULL);
    EXPECT_EQ(a.hi, 0ULL);
    EXPECT_TRUE(a.is_zero());
    EXPECT_FALSE(a.is_negative());
    EXPECT_EQ(a.sgn(), 0);
}

TEST(Int128Signed, ConstructionFromInt64)
{
    int128_t pos(42);
    EXPECT_EQ(pos.lo, 42ULL);
    EXPECT_EQ(pos.hi, 0ULL);
    EXPECT_EQ(pos.sgn(), 1);

    int128_t neg(-42);
    EXPECT_EQ(neg.lo, static_cast<uint64_t>(-42));
    EXPECT_EQ(neg.hi, ~0ULL);
    EXPECT_TRUE(neg.is_negative());
    EXPECT_EQ(neg.sgn(), -1);
}

TEST(Int128Signed, NegationAndAbs)
{
    int128_t a(-42);
    auto n = -a;
    EXPECT_EQ(n.lo, 42ULL);
    EXPECT_EQ(n.hi, 0ULL);

    auto abs_a = a.abs_u();
    EXPECT_EQ(abs_a.lo, 42ULL);
    EXPECT_EQ(abs_a.hi, 0ULL);

    int128_t min_value(0ULL, 0x8000000000000000ULL);
    auto abs_min = min_value.abs_u();
    EXPECT_EQ(abs_min.lo, 0ULL);
    EXPECT_EQ(abs_min.hi, 0x8000000000000000ULL);
}

TEST(Int128Signed, Arithmetic)
{
    EXPECT_TRUE((int128_t(100) + int128_t(-42)) == int128_t(58));
    EXPECT_TRUE((int128_t(42) - int128_t(100)) == int128_t(-58));
    EXPECT_TRUE((int128_t(-100) * int128_t(200)) == int128_t(-20000));
    EXPECT_TRUE((int128_t(-1000) / int128_t(10)) == int128_t(-100));
    EXPECT_TRUE((int128_t(1000) / int128_t(-10)) == int128_t(-100));
    EXPECT_TRUE((int128_t(-1000) / int128_t(-10)) == int128_t(100));
}

TEST(Int128Signed, Comparison)
{
    EXPECT_TRUE(int128_t(-200) < int128_t(-100));
    EXPECT_TRUE(int128_t(-100) < int128_t(0));
    EXPECT_TRUE(int128_t(100) < int128_t(200));
    EXPECT_TRUE(int128_t(200) > int128_t(-200));
    EXPECT_TRUE(int128_t(100) <= int128_t(100));
    EXPECT_TRUE(int128_t(100) >= int128_t(100));
    EXPECT_TRUE(int128_t(100) != int128_t(-100));
}

TEST(Int128Signed, TryAddAndSubOverflow)
{
    int128_t res;
    int128_t max_value(~0ULL, 0x7FFFFFFFFFFFFFFFULL);
    int128_t min_value(0ULL, 0x8000000000000000ULL);

    EXPECT_TRUE(int128_t(100).try_add(int128_t(-42), res));
    EXPECT_TRUE(res == int128_t(58));

    EXPECT_FALSE(max_value.try_add(int128_t(1), res));
    EXPECT_FALSE(min_value.try_sub(int128_t(1), res));

    EXPECT_TRUE(max_value.try_sub(int128_t(1), res));
    EXPECT_EQ(res.lo, ~0ULL - 1ULL);
    EXPECT_EQ(res.hi, 0x7FFFFFFFFFFFFFFFULL);
}

TEST(Int128Signed, TryMulOverflow)
{
    int128_t res;
    int128_t max_value(~0ULL, 0x7FFFFFFFFFFFFFFFULL);
    int128_t min_value(0ULL, 0x8000000000000000ULL);

    EXPECT_TRUE(int128_t(-1000).try_mul(int128_t(1000), res));
    EXPECT_TRUE(res == int128_t(-1000000));

    EXPECT_TRUE(min_value.try_mul(int128_t(1), res));
    EXPECT_TRUE(res == min_value);

    EXPECT_FALSE(max_value.try_mul(int128_t(2), res));
    EXPECT_FALSE(min_value.try_mul(int128_t(-1), res));
}

TEST(Int128Signed, CrossLocalPointsOverflow)
{
    // Reproduces APB_CrossLocalPoints() (the GPU_FUNC in the intersection code)
    // with the values of a case that computes a wrong result:
    //
    //     pa = (157479, 18746978)
    //     pb = (2545264, 22632724)
    //     cx = { hi = 0, lo = 811874791654865148 }
    //     cy = { hi = 5, lo = 16261481622337981762 }
    //     cw = { hi = 0, lo = 4794071381389 }
    //
    //   return (pa.x*cw - cx) * (pb.y*cw - cy)
    //        - (pb.x*cw - cx) * (pa.y*cw - cy);
    //
    // Every sub-expression still fits in a signed 128-bit integer
    // (  ax = -56909224585106817,  by = 7692420390233794,
    //    bx =  11390302508824826548,  ay = -18620851273556547400 )
    // but the products do NOT: |bx * ay| ~ 2.12e38 > INT128_MAX ~ 1.70e38,
    // so the 128-bit evaluation wraps around and the result is off by 2^128.
    const int128_t pa_x(157479), pa_y(18746978);
    const int128_t pb_x(2545264), pb_y(22632724);
    const int128_t cx(811874791654865148ULL, 0ULL);
    const int128_t cy(16261481622337981762ULL, 5ULL);
    const int128_t cw(4794071381389ULL, 0ULL);

    // Intermediates: representable in 128 bits.
    const int128_t ax = pa_x * cw - cx;
    const int128_t by = pb_y * cw - cy;
    const int128_t bx = pb_x * cw - cx;
    const int128_t ay = pa_y * cw - cy;

    EXPECT_TRUE(ax == int128_t(-56909224585106817LL)); // hi = -1
    EXPECT_TRUE(by == int128_t(7692420390233794LL));   // hi = 0
    EXPECT_EQ(bx.lo, 11390302508824826548ULL);        // hi = 0
    EXPECT_EQ(bx.hi, 0ULL);
    EXPECT_EQ(ay.lo, 18272636873862555832ULL);        // hi = -2
    EXPECT_EQ(ay.hi, 0xFFFFFFFFFFFFFFFEULL);

    // 128-bit evaluation, exactly what the kernel does.
    const int128_t result = ax * by - bx * ay;

    // Exact reference: the same expression evaluated in 256-bit, where
    // |bx * ay| is comfortably representable.
    const int256_t ref = int256_t(ax) * int256_t(by) - int256_t(bx) * int256_t(ay);

    EXPECT_TRUE(int256_t(result) == ref);
}

// ============================================================
// int192_t tests
// ============================================================

TEST(Int192Signed, ConstructionAndSign)
{
    int192_t zero;
    EXPECT_TRUE(zero.is_zero());
    EXPECT_EQ(zero.sgn(), 0);

    int192_t pos(42);
    EXPECT_EQ(pos.limb0, 42ULL);
    EXPECT_EQ(pos.limb1, 0ULL);
    EXPECT_EQ(pos.limb2, 0ULL);
    EXPECT_EQ(pos.sgn(), 1);

    int192_t neg(-42);
    EXPECT_EQ(neg.limb0, static_cast<uint64_t>(-42));
    EXPECT_EQ(neg.limb1, ~0ULL);
    EXPECT_EQ(neg.limb2, ~0ULL);
    EXPECT_TRUE(neg.is_negative());
}

TEST(Int192Signed, NegationAndAbs)
{
    int192_t a(-42);
    auto n = -a;
    EXPECT_EQ(n.limb0, 42ULL);
    EXPECT_EQ(n.limb1, 0ULL);
    EXPECT_EQ(n.limb2, 0ULL);

    auto abs_a = a.abs_u();
    EXPECT_EQ(abs_a.lo, 42ULL);
    EXPECT_EQ(abs_a.mi, 0ULL);
    EXPECT_EQ(abs_a.hi, 0ULL);
}

TEST(Int192Signed, ArithmeticAndComparison)
{
    EXPECT_TRUE((int192_t(100) + int192_t(-42)) == int192_t(58));
    EXPECT_TRUE((int192_t(42) - int192_t(100)) == int192_t(-58));
    EXPECT_TRUE((int192_t(-100) * int192_t(200)) == int192_t(-20000));
    EXPECT_TRUE((int192_t(-1000) / int192_t(10)) == int192_t(-100));
    EXPECT_TRUE(int192_t(-200) < int192_t(-100));
    EXPECT_TRUE(int192_t(-100) < int192_t(100));
    EXPECT_TRUE(int192_t(200) > int192_t(100));
}

TEST(Int192Signed, MultiLimbCarryBorrow)
{
    int192_t one(1);
    int192_t two_to_64(0ULL, 1ULL, 0ULL);

    auto sum = int192_t(~0ULL, 0ULL, 0ULL) + one;
    EXPECT_EQ(sum.limb0, 0ULL);
    EXPECT_EQ(sum.limb1, 1ULL);
    EXPECT_EQ(sum.limb2, 0ULL);

    auto diff = two_to_64 - one;
    EXPECT_EQ(diff.limb0, ~0ULL);
    EXPECT_EQ(diff.limb1, 0ULL);
    EXPECT_EQ(diff.limb2, 0ULL);
}

TEST(Int192Signed, TryOperations)
{
    int192_t res;
    int192_t max_value(~0ULL, ~0ULL, 0x7FFFFFFFFFFFFFFFULL);
    int192_t min_value(0ULL, 0ULL, 0x8000000000000000ULL);

    EXPECT_TRUE(int192_t(100).try_add(int192_t(-42), res));
    EXPECT_TRUE(res == int192_t(58));

    EXPECT_FALSE(max_value.try_add(int192_t(1), res));
    EXPECT_FALSE(min_value.try_sub(int192_t(1), res));

    EXPECT_TRUE(int192_t(-1000).try_mul(int192_t(1000), res));
    EXPECT_TRUE(res == int192_t(-1000000));

    EXPECT_FALSE(max_value.try_mul(int192_t(2), res));
    EXPECT_FALSE(min_value.try_mul(int192_t(-1), res));
}

// ============================================================
// int256_t tests
// ============================================================

TEST(Int256Signed, ConstructionAndSign)
{
    int256_t zero;
    EXPECT_TRUE(zero.is_zero());
    EXPECT_EQ(zero.sgn(), 0);

    int256_t pos(42);
    EXPECT_EQ(pos.lo0, 42ULL);
    EXPECT_EQ(pos.lo1, 0ULL);
    EXPECT_EQ(pos.hi0, 0ULL);
    EXPECT_EQ(pos.hi1, 0ULL);
    EXPECT_EQ(pos.sgn(), 1);

    int256_t neg(-42);
    EXPECT_EQ(neg.lo0, static_cast<uint64_t>(-42));
    EXPECT_EQ(neg.lo1, ~0ULL);
    EXPECT_EQ(neg.hi0, ~0ULL);
    EXPECT_EQ(neg.hi1, ~0ULL);
    EXPECT_TRUE(neg.is_negative());
}

TEST(Int256Signed, NegationAndAbs)
{
    int256_t a(-42);
    auto n = -a;
    EXPECT_EQ(n.lo0, 42ULL);
    EXPECT_EQ(n.lo1, 0ULL);
    EXPECT_EQ(n.hi0, 0ULL);
    EXPECT_EQ(n.hi1, 0ULL);

    auto abs_a = a.abs_u();
    EXPECT_EQ(abs_a.lo0, 42ULL);
    EXPECT_EQ(abs_a.lo1, 0ULL);
    EXPECT_EQ(abs_a.hi0, 0ULL);
    EXPECT_EQ(abs_a.hi1, 0ULL);
}

TEST(Int256Signed, ArithmeticAndComparison)
{
    EXPECT_TRUE((int256_t(100) + int256_t(-42)) == int256_t(58));
    EXPECT_TRUE((int256_t(42) - int256_t(100)) == int256_t(-58));
    EXPECT_TRUE((int256_t(-100) * int256_t(200)) == int256_t(-20000));
    EXPECT_TRUE((int256_t(-1000) / int256_t(10)) == int256_t(-100));
    EXPECT_TRUE(int256_t(-200) < int256_t(-100));
    EXPECT_TRUE(int256_t(-100) < int256_t(100));
    EXPECT_TRUE(int256_t(200) > int256_t(100));
}

TEST(Int256Signed, MultiLimbCarryBorrow)
{
    int256_t one(1);
    int256_t two_to_128(0ULL, 0ULL, 1ULL, 0ULL);

    auto sum = int256_t(~0ULL, ~0ULL, 0ULL, 0ULL) + one;
    EXPECT_EQ(sum.lo0, 0ULL);
    EXPECT_EQ(sum.lo1, 0ULL);
    EXPECT_EQ(sum.hi0, 1ULL);
    EXPECT_EQ(sum.hi1, 0ULL);

    auto diff = two_to_128 - one;
    EXPECT_EQ(diff.lo0, ~0ULL);
    EXPECT_EQ(diff.lo1, ~0ULL);
    EXPECT_EQ(diff.hi0, 0ULL);
    EXPECT_EQ(diff.hi1, 0ULL);
}

TEST(Int256Signed, TryOperations)
{
    int256_t res;
    int256_t max_value(~0ULL, ~0ULL, ~0ULL, 0x7FFFFFFFFFFFFFFFULL);
    int256_t min_value(0ULL, 0ULL, 0ULL, 0x8000000000000000ULL);

    EXPECT_TRUE(int256_t(100).try_add(int256_t(-42), res));
    EXPECT_TRUE(res == int256_t(58));

    EXPECT_FALSE(max_value.try_add(int256_t(1), res));
    EXPECT_FALSE(min_value.try_sub(int256_t(1), res));

    EXPECT_TRUE(int256_t(-1000).try_mul(int256_t(1000), res));
    EXPECT_TRUE(res == int256_t(-1000000));

    EXPECT_FALSE(max_value.try_mul(int256_t(2), res));
    EXPECT_FALSE(min_value.try_mul(int256_t(-1), res));
}

