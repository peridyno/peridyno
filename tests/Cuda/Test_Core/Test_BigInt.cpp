#include "gtest/gtest.h"
#include "Basic/BigInt.h"

using namespace dyno;

// ============================================================
// uint192_t tests
// ============================================================

TEST(Uint192, DefaultConstruction)
{
    uint192_t a;
    EXPECT_EQ(a.lo, 0ULL);
    EXPECT_EQ(a.mi, 0ULL);
    EXPECT_EQ(a.hi, 0ULL);
}

TEST(Uint192, Construction)
{
    uint192_t a(1, 2, 3);
    EXPECT_EQ(a.lo, 1ULL);
    EXPECT_EQ(a.mi, 2ULL);
    EXPECT_EQ(a.hi, 3ULL);
}

TEST(Uint192, AdditionSimple)
{
    uint192_t a(100, 0, 0);
    uint192_t b(200, 0, 0);
    auto c = a + b;
    EXPECT_EQ(c.lo, 300ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, AdditionCarry)
{
    uint192_t a(~0ULL, 0, 0);
    uint192_t b(1, 0, 0);
    auto c = a + b;
    EXPECT_EQ(c.lo, 0ULL);
    EXPECT_EQ(c.mi, 1ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, AdditionDoubleCarry)
{
    uint192_t a(~0ULL, ~0ULL, 0);
    uint192_t b(1, 0, 0);
    auto c = a + b;
    EXPECT_EQ(c.lo, 0ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 1ULL);
}

TEST(Uint192, SubtractionSimple)
{
    uint192_t a(300, 0, 0);
    uint192_t b(100, 0, 0);
    auto c = a - b;
    EXPECT_EQ(c.lo, 200ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, SubtractionBorrow)
{
    uint192_t a(0, 1, 0);
    uint192_t b(1, 0, 0);
    auto c = a - b;
    EXPECT_EQ(c.lo, ~0ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, MultiplicationSmall)
{
    uint192_t a(100, 0, 0);
    uint192_t b(200, 0, 0);
    auto c = a * b;
    EXPECT_EQ(c.lo, 20000ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, MultiplicationCarry)
{
    // (2^64 - 1) * 2 = 2^65 - 2
    uint192_t a(~0ULL, 0, 0);
    uint192_t b(2, 0, 0);
    auto c = a * b;
    EXPECT_EQ(c.lo, ~0ULL - 1);
    EXPECT_EQ(c.mi, 1ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, DivisionByUint64)
{
    uint192_t a(1000, 0, 0);
    auto c = a / (uint64_t)10;
    EXPECT_EQ(c.lo, 100ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, DivisionByUint64Large)
{
    // 2^64 / 2 = 2^63
    uint192_t a(0, 1, 0);
    auto c = a / (uint64_t)2;
    EXPECT_EQ(c.lo, 1ULL << 63);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, DivisionByUint128)
{
    uint192_t a(1000, 0, 0);
    uint128_t b(10, 0);
    auto c = a / b;
    EXPECT_EQ(c.lo, 100ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, DivisionByUint192)
{
    uint192_t a(1000, 0, 0);
    uint192_t b(10, 0, 0);
    auto c = a / b;
    EXPECT_EQ(c.lo, 100ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, DivisionByUint192SelfIsSmaller)
{
    uint192_t a(5, 0, 0);
    uint192_t b(10, 0, 0);
    auto c = a / b;
    EXPECT_EQ(c.lo, 0ULL);
    EXPECT_EQ(c.mi, 0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint192, LeftShift)
{
    uint192_t a(1, 0, 0);

    // shift by a non-boundary amount
    auto s32 = a << 32;
    EXPECT_EQ(s32.lo, 1ULL << 32);
    EXPECT_EQ(s32.mi, 0ULL);
    EXPECT_EQ(s32.hi, 0ULL);

    // NOTE: shift==64 hits undefined behavior in source (lo >> 64 for uint64_t),
    // so we skip exact-boundary tests for 64 and 128.

    auto s65 = a << 65;
    EXPECT_EQ(s65.lo, 0ULL);
    EXPECT_EQ(s65.mi, 2ULL);
    EXPECT_EQ(s65.hi, 0ULL);

    auto s129 = a << 129;
    EXPECT_EQ(s129.lo, 0ULL);
    EXPECT_EQ(s129.mi, 0ULL);
    EXPECT_EQ(s129.hi, 2ULL);

    auto d = a << 0;
    EXPECT_EQ(d.lo, 1ULL);

    auto e = a << 192;
    EXPECT_EQ(e.lo, 0ULL);
    EXPECT_EQ(e.mi, 0ULL);
    EXPECT_EQ(e.hi, 0ULL);
}

TEST(Uint192, Comparison)
{
    uint192_t a(100, 0, 0);
    uint192_t b(200, 0, 0);
    uint192_t c(100, 0, 0);
    uint192_t d(0, 1, 0);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a < d);
    EXPECT_TRUE(a >= c);
    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a == b);
}

TEST(Uint192, TryAddSuccess)
{
    uint192_t a(100, 0, 0);
    uint192_t b(200, 0, 0);
    uint192_t res;
    EXPECT_TRUE(a.try_add(b, res));
    EXPECT_EQ(res.lo, 300ULL);
}

TEST(Uint192, TryAddOverflow)
{
    uint192_t a(~0ULL, ~0ULL, ~0ULL);
    uint192_t b(1, 0, 0);
    uint192_t res;
    EXPECT_FALSE(a.try_add(b, res));
}

TEST(Uint192, TrySubSuccess)
{
    uint192_t a(300, 0, 0);
    uint192_t b(100, 0, 0);
    uint192_t res;
    EXPECT_TRUE(a.try_sub(b, res));
    EXPECT_EQ(res.lo, 200ULL);
}

TEST(Uint192, TrySubUnderflow)
{
    uint192_t a(100, 0, 0);
    uint192_t b(200, 0, 0);
    uint192_t res;
    EXPECT_FALSE(a.try_sub(b, res));
}

TEST(Uint192, TryMulSuccess)
{
    uint192_t a(1000000, 0, 0);
    uint192_t b(1000000, 0, 0);
    uint192_t res;
    EXPECT_TRUE(a.try_mul(b, res));
    EXPECT_EQ(res.lo, 1000000000000ULL);
}

TEST(Uint192, SubtractAssign)
{
    uint192_t a(300, 5, 10);
    uint192_t b(100, 2, 3);
    a -= b;
    EXPECT_EQ(a.lo, 200ULL);
    EXPECT_EQ(a.mi, 3ULL);
    EXPECT_EQ(a.hi, 7ULL);
}

// ============================================================
// uint256_t tests
// ============================================================

TEST(Uint256, DefaultConstruction)
{
    uint256_t a;
    EXPECT_EQ(a.lo0, 0ULL);
    EXPECT_EQ(a.lo1, 0ULL);
    EXPECT_EQ(a.hi0, 0ULL);
    EXPECT_EQ(a.hi1, 0ULL);
}

TEST(Uint256, Construction)
{
    uint256_t a(1, 2, 3, 4);
    EXPECT_EQ(a.lo0, 1ULL);
    EXPECT_EQ(a.lo1, 2ULL);
    EXPECT_EQ(a.hi0, 3ULL);
    EXPECT_EQ(a.hi1, 4ULL);
}

TEST(Uint256, AdditionSimple)
{
    uint256_t a(100, 0, 0, 0);
    uint256_t b(200, 0, 0, 0);
    auto c = a + b;
    EXPECT_EQ(c.lo0, 300ULL);
    EXPECT_EQ(c.lo1, 0ULL);
}

TEST(Uint256, AdditionCarry)
{
    uint256_t a(~0ULL, 0, 0, 0);
    uint256_t b(1, 0, 0, 0);
    auto c = a + b;
    EXPECT_EQ(c.lo0, 0ULL);
    EXPECT_EQ(c.lo1, 1ULL);
    EXPECT_EQ(c.hi0, 0ULL);
    EXPECT_EQ(c.hi1, 0ULL);
}

TEST(Uint256, SubtractionSimple)
{
    uint256_t a(300, 0, 0, 0);
    uint256_t b(100, 0, 0, 0);
    auto c = a - b;
    EXPECT_EQ(c.lo0, 200ULL);
}

TEST(Uint256, SubtractionBorrow)
{
    uint256_t a(0, 1, 0, 0);
    uint256_t b(1, 0, 0, 0);
    auto c = a - b;
    EXPECT_EQ(c.lo0, ~0ULL);
    EXPECT_EQ(c.lo1, 0ULL);
}

TEST(Uint256, MultiplicationSmall)
{
    uint256_t a(100, 0, 0, 0);
    uint256_t b(200, 0, 0, 0);
    auto c = a * b;
    EXPECT_EQ(c.lo0, 20000ULL);
    EXPECT_EQ(c.lo1, 0ULL);
    EXPECT_EQ(c.hi0, 0ULL);
    EXPECT_EQ(c.hi1, 0ULL);
}

TEST(Uint256, DivisionByUint64)
{
    uint256_t a(1000, 0, 0, 0);
    auto c = a / (uint64_t)10;
    EXPECT_EQ(c.lo0, 100ULL);
}

TEST(Uint256, DivisionByUint128)
{
    uint256_t a(1000, 0, 0, 0);
    uint128_t b(10, 0);
    auto c = a / b;
    EXPECT_EQ(c.lo0, 100ULL);
    EXPECT_EQ(c.lo1, 0ULL);
}

TEST(Uint256, DivisionByUint192)
{
    uint256_t a(1000, 0, 0, 0);
    uint192_t b(10, 0, 0);
    auto c = a / b;
    EXPECT_EQ(c.lo0, 100ULL);
    EXPECT_EQ(c.lo1, 0ULL);
}

TEST(Uint256, DivisionByUint256)
{
    uint256_t a(1000, 0, 0, 0);
    uint256_t b(10, 0, 0, 0);
    auto c = a / b;
    EXPECT_EQ(c.lo0, 100ULL);
}

TEST(Uint256, DivisionByUint256SelfIsSmaller)
{
    uint256_t a(5, 0, 0, 0);
    uint256_t b(10, 0, 0, 0);
    auto c = a / b;
    EXPECT_EQ(c.lo0, 0ULL);
}

TEST(Uint256, LeftShift)
{
    uint256_t a(1, 0, 0, 0);
    auto b = a << 64;
    EXPECT_EQ(b.lo0, 0ULL);
    EXPECT_EQ(b.lo1, 1ULL);

    auto c = a << 128;
    EXPECT_EQ(c.hi0, 1ULL);

    auto d = a << 192;
    EXPECT_EQ(d.hi1, 1ULL);

    auto e = a << 0;
    EXPECT_EQ(e.lo0, 1ULL);
}

TEST(Uint256, Comparison)
{
    uint256_t a(100, 0, 0, 0);
    uint256_t b(200, 0, 0, 0);
    uint256_t c(100, 0, 0, 0);

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a == b);
}

TEST(Uint256, TryAddSuccess)
{
    uint256_t a(100, 0, 0, 0);
    uint256_t b(200, 0, 0, 0);
    uint256_t res;
    EXPECT_TRUE(a.try_add(b, res));
    EXPECT_EQ(res.lo0, 300ULL);
}

TEST(Uint256, TryAddOverflow)
{
    uint256_t a(~0ULL, ~0ULL, ~0ULL, ~0ULL);
    uint256_t b(1, 0, 0, 0);
    uint256_t res;
    EXPECT_FALSE(a.try_add(b, res));
}

TEST(Uint256, TrySubSuccess)
{
    uint256_t a(300, 0, 0, 0);
    uint256_t b(100, 0, 0, 0);
    uint256_t res;
    EXPECT_TRUE(a.try_sub(b, res));
    EXPECT_EQ(res.lo0, 200ULL);
}

TEST(Uint256, TrySubUnderflow)
{
    uint256_t a(100, 0, 0, 0);
    uint256_t b(200, 0, 0, 0);
    uint256_t res;
    EXPECT_FALSE(a.try_sub(b, res));
}

TEST(Uint256, TryMulSuccess)
{
    uint256_t a(1000000, 0, 0, 0);
    uint256_t b(1000000, 0, 0, 0);
    uint256_t res;
    EXPECT_TRUE(a.try_mul(b, res));
    EXPECT_EQ(res.lo0, 1000000000000ULL);
}

TEST(Uint256, SubtractAssign)
{
    uint256_t a(300, 5, 10, 20);
    uint256_t b(100, 2, 3, 5);
    a -= b;
    EXPECT_EQ(a.lo0, 200ULL);
    EXPECT_EQ(a.lo1, 3ULL);
    EXPECT_EQ(a.hi0, 7ULL);
    EXPECT_EQ(a.hi1, 15ULL);
}

// ============================================================
// ext_sgn_int192_t tests
// ============================================================

TEST(SgnInt192, DefaultConstruction)
{
    ext_sgn_int192_t a;
    EXPECT_EQ(a.limb0, 0ULL);
    EXPECT_EQ(a.limb1, 0ULL);
    EXPECT_EQ(a.limb2, 0ULL);
    EXPECT_EQ(a.neg_mask, 0ULL);
}

TEST(SgnInt192, Construction)
{
    ext_sgn_int192_t a(42, 0, 0, false);
    EXPECT_EQ(a.limb0, 42ULL);
    EXPECT_EQ(a.neg_mask, 0ULL);

    ext_sgn_int192_t b(42, 0, 0, true);
    EXPECT_NE(b.neg_mask, 0ULL);
}

TEST(SgnInt192, Addition)
{
    ext_sgn_int192_t a(100, 0, 0, false);
    ext_sgn_int192_t b(200, 0, 0, false);
    auto c = a + b;
    EXPECT_EQ(c.limb0, 300ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);
}

TEST(SgnInt192, AdditionMixedSigns)
{
    ext_sgn_int192_t a(100, 0, 0, false);
    ext_sgn_int192_t b(300, 0, 0, true);
    auto c = a + b; // 100 + (-300) = -200
    EXPECT_EQ(c.limb0, 200ULL);
    EXPECT_NE(c.neg_mask, 0ULL);
}

TEST(SgnInt192, Subtraction)
{
    ext_sgn_int192_t a(300, 0, 0, false);
    ext_sgn_int192_t b(100, 0, 0, false);
    auto c = a - b;
    EXPECT_EQ(c.limb0, 200ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);

    auto d = b - a; // 100 - 300 = -200
    EXPECT_EQ(d.limb0, 200ULL);
    EXPECT_NE(d.neg_mask, 0ULL);
}

TEST(SgnInt192, Multiplication)
{
    ext_sgn_int192_t a(100, 0, 0, false);
    ext_sgn_int192_t b(200, 0, 0, true);
    auto c = a * b;
    EXPECT_EQ(c.limb0, 20000ULL);
    EXPECT_NE(c.neg_mask, 0ULL);
}

TEST(SgnInt192, Division)
{
    ext_sgn_int192_t a(1000, 0, 0, false);
    ext_sgn_int192_t b(10, 0, 0, false);
    auto c = a / b;
    EXPECT_EQ(c.limb0, 100ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);

    ext_sgn_int192_t d(1000, 0, 0, true);
    auto e = d / b;
    EXPECT_EQ(e.limb0, 100ULL);
    EXPECT_NE(e.neg_mask, 0ULL);
}

TEST(SgnInt192, Sgn)
{
    ext_sgn_int192_t a(100, 0, 0, false);
    EXPECT_EQ(a.sgn(), 1);

    ext_sgn_int192_t b(100, 0, 0, true);
    EXPECT_EQ(b.sgn(), -1);

    ext_sgn_int192_t c;
    EXPECT_EQ(c.sgn(), 0);
}

TEST(SgnInt192, Comparison)
{
    ext_sgn_int192_t a(100, 0, 0, false);
    ext_sgn_int192_t b(200, 0, 0, false);
    ext_sgn_int192_t c(100, 0, 0, true);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(c < a);
    EXPECT_FALSE(b < a);
}

TEST(SgnInt192, Equality)
{
    ext_sgn_int192_t a(100, 0, 0, false);
    ext_sgn_int192_t b(100, 0, 0, false);
    EXPECT_TRUE(a == b);

    ext_sgn_int192_t zero1(0, 0, 0, false);
    ext_sgn_int192_t zero2(0, 0, 0, true);
    EXPECT_TRUE(zero1 == zero2);
}

TEST(SgnInt192, FromInt64Mul)
{
    auto r = ext_sgn_int192_t::from_int64_mul(100, 200, 300);
    EXPECT_EQ(r.limb0, 6000000ULL);
    EXPECT_EQ(r.neg_mask, 0ULL);

    auto r2 = ext_sgn_int192_t::from_int64_mul(-100, 200, 300);
    EXPECT_EQ(r2.limb0, 6000000ULL);
    EXPECT_NE(r2.neg_mask, 0ULL);
}

TEST(SgnInt192, FromInt128Int64Mul)
{
    ext_sgn_int128_t a(100, 0, false);
    auto r = ext_sgn_int192_t::from_int128_int64_mul(a, 200);
    EXPECT_EQ(r.limb0, 20000ULL);
    EXPECT_EQ(r.neg_mask, 0ULL);

    ext_sgn_int128_t b(100, 0, true);
    auto r2 = ext_sgn_int192_t::from_int128_int64_mul(b, 200);
    EXPECT_EQ(r2.limb0, 20000ULL);
    EXPECT_NE(r2.neg_mask, 0ULL);
}

// ============================================================
// ext_sgn_int256_t tests
// ============================================================

TEST(SgnInt256, DefaultConstruction)
{
    ext_sgn_int256_t a;
    EXPECT_EQ(a.lo0, 0ULL);
    EXPECT_EQ(a.lo1, 0ULL);
    EXPECT_EQ(a.hi0, 0ULL);
    EXPECT_EQ(a.hi1, 0ULL);
    EXPECT_EQ(a.neg_mask, 0ULL);
}

TEST(SgnInt256, ConstructionFromInt64)
{
    ext_sgn_int256_t a((int64_t)42);
    EXPECT_EQ(a.lo0, 42ULL);
    EXPECT_EQ(a.neg_mask, 0ULL);

    ext_sgn_int256_t b((int64_t)-42);
    EXPECT_EQ(b.lo0, 42ULL);
    EXPECT_NE(b.neg_mask, 0ULL);
}

TEST(SgnInt256, ConstructionFromSgnInt128)
{
    ext_sgn_int128_t s(100, 200, false);
    ext_sgn_int256_t a(s);
    EXPECT_EQ(a.lo0, 100ULL);
    EXPECT_EQ(a.lo1, 200ULL);
    EXPECT_EQ(a.hi0, 0ULL);
    EXPECT_EQ(a.hi1, 0ULL);
    EXPECT_EQ(a.neg_mask, 0ULL);
}

TEST(SgnInt256, Addition)
{
    ext_sgn_int256_t a(100, 0, 0, 0, false);
    ext_sgn_int256_t b(200, 0, 0, 0, false);
    auto c = a + b;
    EXPECT_EQ(c.lo0, 300ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);
}

TEST(SgnInt256, AdditionMixedSigns)
{
    ext_sgn_int256_t a(100, 0, 0, 0, false);
    ext_sgn_int256_t b(300, 0, 0, 0, true);
    auto c = a + b; // 100 + (-300) = -200
    EXPECT_EQ(c.lo0, 200ULL);
    EXPECT_NE(c.neg_mask, 0ULL);
}

TEST(SgnInt256, Subtraction)
{
    ext_sgn_int256_t a(300, 0, 0, 0, false);
    ext_sgn_int256_t b(100, 0, 0, 0, false);
    auto c = a - b;
    EXPECT_EQ(c.lo0, 200ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);
}

TEST(SgnInt256, Multiplication)
{
    ext_sgn_int256_t a(100, 0, 0, 0, false);
    ext_sgn_int256_t b(200, 0, 0, 0, true);
    auto c = a * b;
    EXPECT_EQ(c.lo0, 20000ULL);
    EXPECT_NE(c.neg_mask, 0ULL);
}

TEST(SgnInt256, Division)
{
    ext_sgn_int256_t a(1000, 0, 0, 0, false);
    ext_sgn_int256_t b(10, 0, 0, 0, false);
    auto c = a / b;
    EXPECT_EQ(c.lo0, 100ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);
}

TEST(SgnInt256, Comparison)
{
    ext_sgn_int256_t a(100, 0, 0, 0, false);
    ext_sgn_int256_t b(200, 0, 0, 0, false);
    ext_sgn_int256_t c(100, 0, 0, 0, true);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(c < a);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
}

TEST(SgnInt256, Equality)
{
    ext_sgn_int256_t a(100, 0, 0, 0, false);
    ext_sgn_int256_t b(100, 0, 0, 0, false);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);

    ext_sgn_int256_t zero1(0, 0, 0, 0, false);
    ext_sgn_int256_t zero2(0, 0, 0, 0, true);
    EXPECT_TRUE(zero1 == zero2);
}

TEST(SgnInt256, TryAddSuccess)
{
    ext_sgn_int256_t a(100, 0, 0, 0, false);
    ext_sgn_int256_t b(200, 0, 0, 0, false);
    ext_sgn_int256_t res;
    EXPECT_TRUE(a.try_add(b, res));
    EXPECT_EQ(res.lo0, 300ULL);
}

TEST(SgnInt256, TrySubSuccess)
{
    ext_sgn_int256_t a(300, 0, 0, 0, false);
    ext_sgn_int256_t b(100, 0, 0, 0, false);
    ext_sgn_int256_t res;
    EXPECT_TRUE(a.try_sub(b, res));
    EXPECT_EQ(res.lo0, 200ULL);
}

TEST(SgnInt256, TryMulSuccess)
{
    ext_sgn_int256_t a(1000, 0, 0, 0, false);
    ext_sgn_int256_t b(1000, 0, 0, 0, false);
    ext_sgn_int256_t res;
    EXPECT_TRUE(a.try_mul(b, res));
    EXPECT_EQ(res.lo0, 1000000ULL);
}

TEST(SgnInt256, FromS192Int64Mul)
{
    ext_sgn_int192_t a(100, 0, 0, false);
    auto r = ext_sgn_int256_t::from_s192_int64_mul(a, 200);
    EXPECT_EQ(r.lo0, 20000ULL);
    EXPECT_EQ(r.neg_mask, 0ULL);

    ext_sgn_int192_t b(100, 0, 0, true);
    auto r2 = ext_sgn_int256_t::from_s192_int64_mul(b, 200);
    EXPECT_EQ(r2.lo0, 20000ULL);
    EXPECT_NE(r2.neg_mask, 0ULL);
}

// ============================================================
// uint512_t tests
// ============================================================

TEST(Uint512, DefaultConstruction)
{
    uint512_t a;
    EXPECT_EQ(a.lo0, 0ULL);
    EXPECT_EQ(a.lo7, 0ULL);
}

TEST(Uint512, Construction4Limb)
{
    uint512_t a(1, 2, 3, 4);
    EXPECT_EQ(a.lo0, 1ULL);
    EXPECT_EQ(a.lo1, 2ULL);
    EXPECT_EQ(a.lo2, 3ULL);
    EXPECT_EQ(a.lo3, 4ULL);
    EXPECT_EQ(a.lo4, 0ULL);
}

TEST(Uint512, AdditionSimple)
{
    uint512_t a(100, 0, 0, 0);
    uint512_t b(200, 0, 0, 0);
    auto c = a + b;
    EXPECT_EQ(c.lo0, 300ULL);
}

TEST(Uint512, AdditionCarry)
{
    uint512_t a(~0ULL, 0, 0, 0);
    uint512_t b(1, 0, 0, 0);
    auto c = a + b;
    EXPECT_EQ(c.lo0, 0ULL);
    EXPECT_EQ(c.lo1, 1ULL);
}

TEST(Uint512, SubtractionSimple)
{
    uint512_t a(300, 0, 0, 0);
    uint512_t b(100, 0, 0, 0);
    auto c = a - b;
    EXPECT_EQ(c.lo0, 200ULL);
}

TEST(Uint512, MultiplicationSmall)
{
    uint512_t a(100, 0, 0, 0);
    uint512_t b(200, 0, 0, 0);
    auto c = a * b;
    EXPECT_EQ(c.lo0, 20000ULL);
}

TEST(Uint512, Comparison)
{
    uint512_t a(100, 0, 0, 0);
    uint512_t b(200, 0, 0, 0);
    uint512_t c(100, 0, 0, 0);

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b > a);
}

TEST(Uint512, TryAddSuccess)
{
    uint512_t a(100, 0, 0, 0);
    uint512_t b(200, 0, 0, 0);
    uint512_t res;
    EXPECT_TRUE(a.try_add(b, res));
    EXPECT_EQ(res.lo0, 300ULL);
}

TEST(Uint512, TryAddOverflow)
{
    uint512_t a(~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL);
    uint512_t b(1, 0, 0, 0);
    uint512_t res;
    EXPECT_FALSE(a.try_add(b, res));
}

TEST(Uint512, TrySubSuccess)
{
    uint512_t a(300, 0, 0, 0);
    uint512_t b(100, 0, 0, 0);
    uint512_t res;
    EXPECT_TRUE(a.try_sub(b, res));
    EXPECT_EQ(res.lo0, 200ULL);
}

TEST(Uint512, TrySubUnderflow)
{
    uint512_t a(100, 0, 0, 0);
    uint512_t b(200, 0, 0, 0);
    uint512_t res;
    EXPECT_FALSE(a.try_sub(b, res));
}

TEST(Uint512, TryMulSuccess)
{
    uint512_t a(1000000, 0, 0, 0);
    uint512_t b(1000000, 0, 0, 0);
    uint512_t res;
    EXPECT_TRUE(a.try_mul(b, res));
    EXPECT_EQ(res.lo0, 1000000000000ULL);
}

// ============================================================
// ext_sgn_int512_t tests
// ============================================================

TEST(SgnInt512, DefaultConstruction)
{
    ext_sgn_int512_t a;
    EXPECT_EQ(a.abs_val.lo0, 0ULL);
    EXPECT_EQ(a.neg_mask, 0ULL);
}

TEST(SgnInt512, ConstructionFromSgnInt256)
{
    ext_sgn_int256_t s(100, 0, 0, 0, true);
    ext_sgn_int512_t a(s);
    EXPECT_EQ(a.abs_val.lo0, 100ULL);
    EXPECT_NE(a.neg_mask, 0ULL);
}

TEST(SgnInt512, Addition)
{
    ext_sgn_int512_t a(uint512_t(100, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(200, 0, 0, 0), false);
    auto c = a + b;
    EXPECT_EQ(c.abs_val.lo0, 300ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);
}

TEST(SgnInt512, AdditionMixedSigns)
{
    ext_sgn_int512_t a(uint512_t(100, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(300, 0, 0, 0), true);
    auto c = a + b; // 100 + (-300) = -200
    EXPECT_EQ(c.abs_val.lo0, 200ULL);
    EXPECT_NE(c.neg_mask, 0ULL);
}

TEST(SgnInt512, Subtraction)
{
    ext_sgn_int512_t a(uint512_t(300, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(100, 0, 0, 0), false);
    auto c = a - b;
    EXPECT_EQ(c.abs_val.lo0, 200ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);
}

TEST(SgnInt512, Multiplication)
{
    ext_sgn_int512_t a(uint512_t(100, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(200, 0, 0, 0), true);
    auto c = a * b;
    EXPECT_EQ(c.abs_val.lo0, 20000ULL);
    EXPECT_NE(c.neg_mask, 0ULL);
}

TEST(SgnInt512, Comparison)
{
    ext_sgn_int512_t a(uint512_t(100, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(200, 0, 0, 0), false);
    ext_sgn_int512_t c(uint512_t(100, 0, 0, 0), true);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(c < a);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
}

TEST(SgnInt512, Equality)
{
    ext_sgn_int512_t a(uint512_t(100, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(100, 0, 0, 0), false);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);

    ext_sgn_int512_t zero1(uint512_t(), false);
    ext_sgn_int512_t zero2(uint512_t(), true);
    EXPECT_TRUE(zero1 == zero2);
}

TEST(SgnInt512, TryAddSuccess)
{
    ext_sgn_int512_t a(uint512_t(100, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(200, 0, 0, 0), false);
    ext_sgn_int512_t res;
    EXPECT_TRUE(a.try_add(b, res));
    EXPECT_EQ(res.abs_val.lo0, 300ULL);
}

TEST(SgnInt512, TrySubSuccess)
{
    ext_sgn_int512_t a(uint512_t(300, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(100, 0, 0, 0), false);
    ext_sgn_int512_t res;
    EXPECT_TRUE(a.try_sub(b, res));
    EXPECT_EQ(res.abs_val.lo0, 200ULL);
}

TEST(SgnInt512, TryMulSuccess)
{
    ext_sgn_int512_t a(uint512_t(1000, 0, 0, 0), false);
    ext_sgn_int512_t b(uint512_t(1000, 0, 0, 0), false);
    ext_sgn_int512_t res;
    EXPECT_TRUE(a.try_mul(b, res));
    EXPECT_EQ(res.abs_val.lo0, 1000000ULL);
}

// ============================================================
// Cross-type arithmetic verification
// ============================================================

TEST(BigIntCross, Uint192DivUint128Consistency)
{
    // 2000000 / 100 = 20000 — verify via both uint128 and uint64 paths
    uint192_t a(2000000, 0, 0);
    uint128_t b128(100, 0);
    auto r128 = a / b128;
    auto r64 = a / (uint64_t)100;
    EXPECT_EQ(r128.lo, r64.lo);
    EXPECT_EQ(r128.mi, r64.mi);
    EXPECT_EQ(r128.hi, r64.hi);
}

TEST(BigIntCross, Uint256MulIdentity)
{
    uint256_t a(12345678, 0, 0, 0);
    uint256_t one(1, 0, 0, 0);
    auto r = a * one;
    EXPECT_EQ(r.lo0, 12345678ULL);
    EXPECT_EQ(r.lo1, 0ULL);
    EXPECT_EQ(r.hi0, 0ULL);
    EXPECT_EQ(r.hi1, 0ULL);
}

TEST(BigIntCross, Uint192MulDivRoundtrip)
{
    // a * b / b == a (for small values)
    uint192_t a(999, 0, 0);
    uint192_t b(7, 0, 0);
    auto product = a * b;
    auto quotient = product / b;
    EXPECT_EQ(quotient.lo, 999ULL);
    EXPECT_EQ(quotient.mi, 0ULL);
    EXPECT_EQ(quotient.hi, 0ULL);
}

TEST(BigIntCross, SgnInt256AddSubInverse)
{
    ext_sgn_int256_t a(12345, 0, 0, 0, false);
    ext_sgn_int256_t b(67890, 0, 0, 0, true);
    auto sum = a + b;
    auto original = sum - b;
    EXPECT_EQ(original.lo0, a.lo0);
    EXPECT_EQ(original.neg_mask, a.neg_mask);
}