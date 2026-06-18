#include "gtest/gtest.h"
#include "Basic/BigInt128.h"

using namespace dyno;

// ============================================================
// uint128_t tests
// ============================================================

TEST(Uint128, DefaultConstruction)
{
    uint128_t a{};
    EXPECT_EQ(a.lo, 0ULL);
    EXPECT_EQ(a.hi, 0ULL);
}

TEST(Uint128, Construction)
{
    uint128_t a(42, 0);
    EXPECT_EQ(a.lo, 42ULL);
    EXPECT_EQ(a.hi, 0ULL);

    uint128_t b(0, 1);
    EXPECT_EQ(b.lo, 0ULL);
    EXPECT_EQ(b.hi, 1ULL);
}

TEST(Uint128, AdditionSimple)
{
    uint128_t a(100, 0);
    uint128_t b(200, 0);
    uint128_t c = a + b;
    EXPECT_EQ(c.lo, 300ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint128, AdditionCarry)
{
    uint128_t a(~0ULL, 0);
    uint128_t b(1, 0);
    uint128_t c = a + b;
    EXPECT_EQ(c.lo, 0ULL);
    EXPECT_EQ(c.hi, 1ULL);
}

TEST(Uint128, SubtractionSimple)
{
    uint128_t a(300, 0);
    uint128_t b(100, 0);
    uint128_t c = a - b;
    EXPECT_EQ(c.lo, 200ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint128, SubtractionBorrow)
{
    uint128_t a(0, 1);
    uint128_t b(1, 0);
    uint128_t c = a - b;
    EXPECT_EQ(c.lo, ~0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint128, MultiplicationSmall)
{
    uint128_t a(100, 0);
    uint128_t b(200, 0);
    uint128_t c = a * b;
    EXPECT_EQ(c.lo, 20000ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint128, MultiplicationLarge)
{
    // (2^64 - 1) * 2 = 2^65 - 2
    uint128_t a(~0ULL, 0);
    uint128_t b(2, 0);
    uint128_t c = a * b;
    EXPECT_EQ(c.lo, ~0ULL - 1);
    EXPECT_EQ(c.hi, 1ULL);
}

TEST(Uint128, DivisionByUint64)
{
    uint128_t a(1000, 0);
    uint128_t c = a / (uint64_t)10;
    EXPECT_EQ(c.lo, 100ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint128, DivisionByUint64Large)
{
    // 2^64 / 2 = 2^63
    uint128_t a(0, 1);
    uint128_t c = a / (uint64_t)2;
    EXPECT_EQ(c.lo, 1ULL << 63);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint128, DivisionByUint128)
{
    uint128_t a(1000, 0);
    uint128_t b(10, 0);
    uint128_t c = a / b;
    EXPECT_EQ(c.lo, 100ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint128, DivisionByUint128Large)
{
    // (2^128 - 1) / (2^64) should be (2^64 - 1)
    uint128_t a(~0ULL, ~0ULL);
    uint128_t b(0, 1);
    uint128_t c = a / b;
    EXPECT_EQ(c.lo, ~0ULL);
    EXPECT_EQ(c.hi, 0ULL);
}

TEST(Uint128, ModuloUint64)
{
    uint128_t a(107, 0);
    uint64_t r = a % (uint64_t)10;
    EXPECT_EQ(r, 7ULL);
}

TEST(Uint128, ModuloUint128)
{
    uint128_t a(107, 0);
    uint128_t b(10, 0);
    uint128_t r = a % b;
    EXPECT_EQ(r.lo, 7ULL);
    EXPECT_EQ(r.hi, 0ULL);
}

TEST(Uint128, LeftShift)
{
    uint128_t a(1, 0);
    uint128_t b = a << 64;
    EXPECT_EQ(b.lo, 0ULL);
    EXPECT_EQ(b.hi, 1ULL);

    uint128_t c = a << 0;
    EXPECT_EQ(c.lo, 1ULL);
    EXPECT_EQ(c.hi, 0ULL);

    uint128_t d = a << 32;
    EXPECT_EQ(d.lo, 1ULL << 32);
    EXPECT_EQ(d.hi, 0ULL);
}

TEST(Uint128, RightShift)
{
    uint128_t a(0, 1);
    uint128_t b = a >> 64;
    EXPECT_EQ(b.lo, 1ULL);
    EXPECT_EQ(b.hi, 0ULL);

    uint128_t c = a >> 0;
    EXPECT_EQ(c.lo, 0ULL);
    EXPECT_EQ(c.hi, 1ULL);
}

TEST(Uint128, Comparison)
{
    uint128_t a(100, 0);
    uint128_t b(200, 0);
    uint128_t c(100, 0);
    uint128_t d(0, 1);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a < d);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(a >= c);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a == b);
}

TEST(Uint128, TryAddSuccess)
{
    uint128_t a(100, 0);
    uint128_t b(200, 0);
    uint128_t res;
    EXPECT_TRUE(a.try_add(b, res));
    EXPECT_EQ(res.lo, 300ULL);
    EXPECT_EQ(res.hi, 0ULL);
}

TEST(Uint128, TryAddOverflow)
{
    uint128_t a(~0ULL, ~0ULL);
    uint128_t b(1, 0);
    uint128_t res;
    EXPECT_FALSE(a.try_add(b, res));
}

TEST(Uint128, TrySubSuccess)
{
    uint128_t a(300, 0);
    uint128_t b(100, 0);
    uint128_t res;
    EXPECT_TRUE(a.try_sub(b, res));
    EXPECT_EQ(res.lo, 200ULL);
}

TEST(Uint128, TrySubUnderflow)
{
    uint128_t a(100, 0);
    uint128_t b(200, 0);
    uint128_t res;
    EXPECT_FALSE(a.try_sub(b, res));
}

TEST(Uint128, TryMulSuccess)
{
    uint128_t a(1000000, 0);
    uint128_t b(1000000, 0);
    uint128_t res;
    EXPECT_TRUE(a.try_mul(b, res));
    EXPECT_EQ(res.lo, 1000000000000ULL);
    EXPECT_EQ(res.hi, 0ULL);
}

TEST(Uint128, FromUint64Mul)
{
    auto r = uint128_t::from_uint64_mul(~0ULL, ~0ULL);
    // (2^64-1)^2 = 2^128 - 2^65 + 1
    // hi = 2^64 - 2, lo = 1
    EXPECT_EQ(r.lo, 1ULL);
    EXPECT_EQ(r.hi, ~0ULL - 1);
}

// ============================================================
// ext_sgn_int128_t tests
// ============================================================

TEST(SgnInt128, DefaultConstruction)
{
    ext_sgn_int128_t a;
    EXPECT_EQ(a.lo, 0ULL);
    EXPECT_EQ(a.hi, 0ULL);
    EXPECT_EQ(a.neg_mask, 0ULL);
}

TEST(SgnInt128, FromInt64)
{
    ext_sgn_int128_t a(42);
    EXPECT_EQ(a.lo, 42ULL);
    EXPECT_EQ(a.hi, 0ULL);
    EXPECT_EQ(a.neg_mask, 0ULL);

    ext_sgn_int128_t b(-42);
    EXPECT_EQ(b.lo, 42ULL);
    EXPECT_EQ(b.hi, 0ULL);
    EXPECT_NE(b.neg_mask, 0ULL);
}

TEST(SgnInt128, FromInt64Mul)
{
    auto r = ext_sgn_int128_t::from_int64_mul(100, 200);
    EXPECT_EQ(r.lo, 20000ULL);
    EXPECT_EQ(r.hi, 0ULL);
    EXPECT_EQ(r.neg_mask, 0ULL);

    auto r2 = ext_sgn_int128_t::from_int64_mul(-100, 200);
    EXPECT_EQ(r2.lo, 20000ULL);
    EXPECT_NE(r2.neg_mask, 0ULL);

    auto r3 = ext_sgn_int128_t::from_int64_mul(-100, -200);
    EXPECT_EQ(r3.lo, 20000ULL);
    EXPECT_EQ(r3.neg_mask, 0ULL);
}

TEST(SgnInt128, Addition)
{
    ext_sgn_int128_t a(42, 0, false);
    ext_sgn_int128_t b(58, 0, false);
    auto c = a + b;
    EXPECT_EQ(c.lo, 100ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);

    ext_sgn_int128_t d(100, 0, true);
    auto e = a + d; // 42 + (-100) = -58
    EXPECT_EQ(e.lo, 58ULL);
    EXPECT_NE(e.neg_mask, 0ULL);
}

TEST(SgnInt128, Subtraction)
{
    ext_sgn_int128_t a(100, 0, false);
    ext_sgn_int128_t b(42, 0, false);
    auto c = a - b; // 100 - 42 = 58
    EXPECT_EQ(c.lo, 58ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);

    auto d = b - a; // 42 - 100 = -58
    EXPECT_EQ(d.lo, 58ULL);
    EXPECT_NE(d.neg_mask, 0ULL);
}

TEST(SgnInt128, Multiplication)
{
    ext_sgn_int128_t a(100, 0, false);
    ext_sgn_int128_t b(200, 0, true);
    auto c = a * b; // 100 * (-200) = -20000
    EXPECT_EQ(c.lo, 20000ULL);
    EXPECT_NE(c.neg_mask, 0ULL);

    ext_sgn_int128_t d(100, 0, true);
    auto e = d * b; // (-100) * (-200) = 20000
    EXPECT_EQ(e.lo, 20000ULL);
    EXPECT_EQ(e.neg_mask, 0ULL);
}

TEST(SgnInt128, Division)
{
    ext_sgn_int128_t a(1000, 0, false);
    ext_sgn_int128_t b(10, 0, false);
    auto c = a / b;
    EXPECT_EQ(c.lo, 100ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);

    ext_sgn_int128_t d(1000, 0, true);
    auto e = d / b; // -1000 / 10 = -100
    EXPECT_EQ(e.lo, 100ULL);
    EXPECT_NE(e.neg_mask, 0ULL);
}

TEST(SgnInt128, Modulo)
{
    ext_sgn_int128_t a(107, 0, false);
    ext_sgn_int128_t b(10, 0, false);
    auto c = a % b;
    EXPECT_EQ(c.lo, 7ULL);
    EXPECT_EQ(c.neg_mask, 0ULL);
}

TEST(SgnInt128, Comparison)
{
    ext_sgn_int128_t a(100, 0, false);
    ext_sgn_int128_t b(200, 0, false);
    ext_sgn_int128_t c(100, 0, true);

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(c < a);   // -100 < 100
    EXPECT_TRUE(c < b);   // -100 < 200
    EXPECT_FALSE(a < c);
}

TEST(SgnInt128, Equality)
{
    ext_sgn_int128_t a(100, 0, false);
    ext_sgn_int128_t b(100, 0, false);
    ext_sgn_int128_t c(100, 0, true);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);

    ext_sgn_int128_t zero1(0, 0, false);
    ext_sgn_int128_t zero2(0, 0, true);
    EXPECT_TRUE(zero1 == zero2);
}

TEST(SgnInt128, Negation)
{
    ext_sgn_int128_t a(100, 0, false);
    auto b = -a;
    EXPECT_EQ(b.lo, 100ULL);
    EXPECT_NE(b.neg_mask, 0ULL);

    ext_sgn_int128_t zero(0, 0, false);
    auto nz = -zero;
    EXPECT_EQ(nz.lo, 0ULL);
    EXPECT_EQ(nz.neg_mask, 0ULL);
}

TEST(SgnInt128, Sgn)
{
    ext_sgn_int128_t a(100, 0, false);
    EXPECT_EQ(a.sgn(), 1);

    ext_sgn_int128_t b(100, 0, true);
    EXPECT_EQ(b.sgn(), -1);

    ext_sgn_int128_t c(0, 0, false);
    EXPECT_EQ(c.sgn(), 0);
}

TEST(SgnInt128, TryAddSuccess)
{
    ext_sgn_int128_t a(100, 0, false);
    ext_sgn_int128_t b(200, 0, false);
    ext_sgn_int128_t res;
    EXPECT_TRUE(a.try_add(b, res));
    EXPECT_EQ(res.lo, 300ULL);
}

TEST(SgnInt128, TrySubSuccess)
{
    ext_sgn_int128_t a(300, 0, false);
    ext_sgn_int128_t b(100, 0, false);
    ext_sgn_int128_t res;
    EXPECT_TRUE(a.try_sub(b, res));
    EXPECT_EQ(res.lo, 200ULL);
}

TEST(SgnInt128, TryMulSuccess)
{
    ext_sgn_int128_t a(1000, 0, false);
    ext_sgn_int128_t b(1000, 0, false);
    ext_sgn_int128_t res;
    EXPECT_TRUE(a.try_mul(b, res));
    EXPECT_EQ(res.lo, 1000000ULL);
}

TEST(SgnInt128, FromInt32Mul)
{
    auto r = ext_sgn_int128_t::from_int32_mul(100, 200, 300);
    EXPECT_EQ(r.lo, 6000000ULL);
    EXPECT_EQ(r.neg_mask, 0ULL);

    auto r2 = ext_sgn_int128_t::from_int32_mul(-100, 200, 300);
    EXPECT_EQ(r2.lo, 6000000ULL);
    EXPECT_NE(r2.neg_mask, 0ULL);
}

TEST(SgnInt128, Abs)
{
    ext_sgn_int128_t a(42, 0, true);
    auto abs_a = a.abs();
    EXPECT_EQ(abs_a.lo, 42ULL);
    EXPECT_EQ(abs_a.hi, 0ULL);
}