// Random and extreme BigInt -> double conversion cases.
//
// The expected values were computed offline with Python's arbitrary precision
// integers/Fractions and rounded once to the nearest double, so they are correctly
// rounded, implementation-independent references.  Every row is checked against
// to_double, to_double_scaled_by and to_double_divided_by; rows whose resolution is
// exactly 1 therefore also pin the two scaled paths to the plain conversion.

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "Basic/BigIntFloat.h"

using namespace dyno;

namespace
{
// Compares `actual` against the correctly rounded reference `expected`, allowing
// `max_ulps` units in the last place of `expected`.  The conversion runs in `double`,
// the only type available on the CUDA device; one ulp of slack is measured, two is
// the margin we allow here.
void expect_close_double(double actual, double expected, double max_ulps,
                         const std::string& context)
{
    if (std::isinf(expected))
    {
        EXPECT_EQ(actual, expected) << context;
        return;
    }
    ASSERT_TRUE(std::isfinite(actual)) << context;

    if (expected == 0.0)
    {
        EXPECT_EQ(actual, 0.0) << context;
        return;
    }

    const double ulp = std::fabs(expected) - std::nextafter(std::fabs(expected), 0.0);
    EXPECT_LE(std::fabs(actual - expected), max_ulps * ulp)
        << context << ": actual=" << std::setprecision(17) << actual
        << " expected=" << expected;
}
} // namespace

// ---------------------------------------------------------------------------
// uint128_t
// ---------------------------------------------------------------------------

struct Uint128Case
{
    uint64_t lo;
    uint64_t hi;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static uint128_t make_value(const Uint128Case& c) { return uint128_t(c.lo, c.hi); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
static const Uint128Case kUint128Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.a8f943a3d4000p+39 /* 912624046570 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.1d6483a46c2a3p+13 /* 9132.5642784548163 */, 0x1.0000000000000p+128 /* 3.4028236692093846e+38 */, 0x1.1d6483a46c2a3p+141 /* 3.1076505887302174e+42 */, 0x1.cb44df9917e64p+114 /* 3.7260330893451156e+34 */},
    {0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+127 /* 1.7014118346046923e+38 */, 0x1.0000000000000p+127 /* 1.7014118346046923e+38 */, 0x1.0000000000000p+127 /* 1.7014118346046923e+38 */},
    {0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0624dd2f1a9fcp-10 /* 0.001 */, 0x1.0000000000000p+127 /* 1.7014118346046923e+38 */, 0x1.0624dd2f1a9fcp+117 /* 1.7014118346046924e+35 */, 0x1.f400000000000p+136 /* 1.7014118346046923e+41 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x1.34a456d5cfaadp+10 /* 1234.5678 */, 0x1.0000000000000p+0 /* 1 */, 0x1.34a456d5cfaadp+10 /* 1234.5678 */, 0x1.a8ac5e5c399b5p-11 /* 0.00081000006642000541 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */},
    {0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0x1.999999999999ap-4 /* 0.10000000000000001 */, 0x1.0000000000000p+127 /* 1.7014118346046923e+38 */, 0x1.999999999999ap+123 /* 1.7014118346046924e+37 */, 0x1.4000000000000p+130 /* 1.7014118346046923e+39 */},
    {0x0000000000000000ULL, 0x0123456789ABCDEFULL, 0x1.0000000000000p+149 /* 7.1362384635297994e+44 */, 0x1.23456789abcdfp+120 /* 1.5123660752041709e+36 */, 0x1.23456789abcdfp+269 /* 1.0792604956809606e+81 */, 0x1.23456789abcdfp-29 /* 2.1192762586805556e-09 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x1.0b6db948cf95cp+10 /* 1069.7144338633725 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.30463a59ab551p+66 /* 8.770110397080327e+19 */, 0x1.16d2d6bc79c0ap+46 /* 76642444582512.156 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.e1e1e1e1e1e1ep+127 /* 3.2026575710205973e+38 */, 0x1.e1e1e1e1e1e1ep+127 /* 3.2026575710205973e+38 */, 0x1.e1e1e1e1e1e1ep+127 /* 3.2026575710205973e+38 */},
    {0xB842879B844D99F8ULL, 0x6360DF67A8A188C4ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.8d837d9ea2862p+126 /* 1.3209656327115713e+38 */, 0x1.8d837d9ea2862p+126 /* 1.3209656327115713e+38 */, 0x1.8d837d9ea2862p+126 /* 1.3209656327115713e+38 */},
    {0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x1.294f15760fb5dp+15 /* 38055.541916361755 */, 0x1.0000000000000p+64 /* 1.8446744073709552e+19 */, 0x1.294f15760fb5dp+79 /* 7.0200084231745164e+23 */, 0x1.b8dc7eb4938f4p+48 /* 484732134790031.25 */},
    {0xDC70AF0EA958CF5AULL, 0x93E78E6D2C94AD0DULL, 0x1.0000000000000p+0 /* 1 */, 0x1.27cf1cda59296p+127 /* 1.965988247065347e+38 */, 0x1.27cf1cda59296p+127 /* 1.965988247065347e+38 */, 0x1.27cf1cda59296p+127 /* 1.965988247065347e+38 */},
    {0x2AE412C808D4C133ULL, 0x5436A95A8093DA01ULL, 0x1.07cfffbcc29f3p+15 /* 33767.999487001354 */, 0x1.50daa56a024f7p+126 /* 1.1193897057384429e+38 */, 0x1.5b22513dd484bp+141 /* 3.7799551009130335e+42 */, 0x1.46e0e8a64a664p+111 /* 3.3149423203744728e+33 */},
    {0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x1.0000000000000p-121 /* 3.76158192263132e-37 */, 0x1.0000000000000p+64 /* 1.8446744073709552e+19 */, 0x1.0000000000000p-57 /* 6.9388939039072284e-18 */, 0x1.0000000000000p+185 /* 4.9039857307708443e+55 */},
    {0x0000000000003F5CULL, 0x0000000000000006ULL, 0x1.0000000000000p+697 /* 6.5751698769354669e+209 */, 0x1.8000000000001p+66 /* 1.1068046444225733e+20 */, 0x1.8000000000001p+763 /* 7.2774285576595742e+229 */, 0x1.8000000000001p-631 /* 1.6833095800384538e-190 */},
};

constexpr std::size_t kUint128CaseCount = sizeof(kUint128Cases) / sizeof(kUint128Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kUint128UlpSlack = 2.0;

TEST(BigIntFloatUint128, ToDouble)
{
    for (std::size_t i = 0; i < kUint128CaseCount; ++i)
    {
        const Uint128Case& c = kUint128Cases[i];
        const uint128_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kUint128UlpSlack,
                           std::string("uint128_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatUint128, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kUint128CaseCount; ++i)
    {
        const Uint128Case& c = kUint128Cases[i];
        const uint128_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kUint128UlpSlack,
                           std::string("uint128_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatUint128, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kUint128CaseCount; ++i)
    {
        const Uint128Case& c = kUint128Cases[i];
        const uint128_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kUint128UlpSlack,
                           std::string("uint128_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// uint192_t
// ---------------------------------------------------------------------------

struct Uint192Case
{
    uint64_t lo;
    uint64_t mi;
    uint64_t hi;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static uint192_t make_value(const Uint192Case& c) { return uint192_t(c.lo, c.mi, c.hi); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
static const Uint192Case kUint192Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.54db9e03e5d8dp-11 /* 0.00065013481115532153 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.0000000000000p+7 /* 128 */, 0x1.0000000000000p+192 /* 6.2771017353866808e+57 */, 0x1.0000000000000p+199 /* 8.0346902212949514e+59 */, 0x1.0000000000000p+185 /* 4.9039857307708443e+55 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p-46 /* 1.4210854715202004e-14 */, 0x1.0000000000000p+191 /* 3.1385508676933404e+57 */, 0x1.0000000000000p+145 /* 4.4601490397061246e+43 */, 0x1.0000000000000p+237 /* 2.2085588309729804e+71 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+191 /* 3.1385508676933404e+57 */, 0x1.0000000000000p+191 /* 3.1385508676933404e+57 */, 0x1.0000000000000p+191 /* 3.1385508676933404e+57 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p-272 /* 1.3177747429038154e-82 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p-272 /* 1.3177747429038154e-82 */, 0x1.0000000000000p+272 /* 7.5885503602567542e+81 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p-457 /* 2.6871504430268355e-138 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p-457 /* 2.6871504430268355e-138 */, 0x1.0000000000000p+457 /* 3.7214142683935073e+137 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0x1.0000000000000p-26 /* 1.4901161193847656e-08 */, 0x1.0000000000000p+191 /* 3.1385508676933404e+57 */, 0x1.0000000000000p+165 /* 4.6768052394588893e+49 */, 0x1.0000000000000p+217 /* 2.1062458333711437e+65 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0123456789ABCDEFULL, 0x1.e848000000000p+19 /* 1000000 */, 0x1.23456789abcdfp+184 /* 2.7898229935051914e+55 */, 0x1.15c71c71c71c7p+204 /* 2.7898229935051914e+61 */, 0x1.316b7e5807ca5p+164 /* 2.7898229935051913e+49 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.29fde5d1a0000p+35 /* 39995780749 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.530c6631a2b3cp+91 /* 3.2790752511333435e+27 */, 0x1.f473e73a0dc81p+20 /* 2049854.4516732993 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.e1e1e1e1e1e1ep+191 /* 5.907860456834523e+57 */, 0x1.e1e1e1e1e1e1ep+191 /* 5.907860456834523e+57 */, 0x1.e1e1e1e1e1e1ep+191 /* 5.907860456834523e+57 */},
    {0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.6c95ec774c000p+39 /* 782941698982 */, 0x1.0000000000000p+192 /* 6.2771017353866808e+57 */, 0x1.6c95ec774c000p+231 /* 4.9146046973865084e+69 */, 0x1.67826e6c46e8bp+152 /* 8.0173296984287876e+45 */},
    {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x1.0000000000000p+35 /* 34359738368 */, 0x1.0000000000000p+128 /* 3.4028236692093846e+38 */, 0x1.0000000000000p+163 /* 1.1692013098647223e+49 */, 0x1.0000000000000p+93 /* 9.9035203142830422e+27 */},
    {0x0000000000000001ULL, 0x8C9B6690944A8E70ULL, 0x0000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.1936cd2128952p+127 /* 1.8689880568349002e+38 */, 0x1.1936cd2128952p+127 /* 1.8689880568349002e+38 */, 0x1.1936cd2128952p+127 /* 1.8689880568349002e+38 */},
    {0x40E6EA63279989F7ULL, 0x7AFD84482B100702ULL, 0xDA5CC197A74D7C40ULL, 0x1.bd8c0f87e4000p+38 /* 478402568697 */, 0x1.b4b9832f4e9b0p+191 /* 5.3542287274149734e+57 */, 0x1.7c0ab47dd57dcp+230 /* 2.5614767765865924e+69 */, 0x1.f5dc9776c55ecp+152 /* 1.1191889587879942e+46 */},
    {0x85C41E89A1F0756FULL, 0x26198BCA7880559DULL, 0x19B65E1538C82D37ULL, 0x1.0000000000000p-1 /* 0.5 */, 0x1.9b65e1538c82dp+188 /* 6.304655537148077e+56 */, 0x1.9b65e1538c82dp+187 /* 3.1523277685740385e+56 */, 0x1.9b65e1538c82dp+189 /* 1.2609311074296154e+57 */},
    {0xDAFDBFA6D292BF62ULL, 0x00000000000041DCULL, 0x0000000021E4C846ULL, 0x1.0000000000000p-603 /* 3.0123998313786051e-182 */, 0x1.0f26423000000p+157 /* 1.9349871161940372e+47 */, 0x1.0f26423000000p-446 /* 5.8289548625426912e-135 */, 0x1.0f26423000000p+760 /* 6.4234073313850339e+228 */},
};

constexpr std::size_t kUint192CaseCount = sizeof(kUint192Cases) / sizeof(kUint192Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kUint192UlpSlack = 2.0;

TEST(BigIntFloatUint192, ToDouble)
{
    for (std::size_t i = 0; i < kUint192CaseCount; ++i)
    {
        const Uint192Case& c = kUint192Cases[i];
        const uint192_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kUint192UlpSlack,
                           std::string("uint192_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatUint192, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kUint192CaseCount; ++i)
    {
        const Uint192Case& c = kUint192Cases[i];
        const uint192_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kUint192UlpSlack,
                           std::string("uint192_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatUint192, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kUint192CaseCount; ++i)
    {
        const Uint192Case& c = kUint192Cases[i];
        const uint192_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kUint192UlpSlack,
                           std::string("uint192_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// uint256_t
// ---------------------------------------------------------------------------

struct Uint256Case
{
    uint64_t lo0;
    uint64_t lo1;
    uint64_t hi0;
    uint64_t hi1;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static uint256_t make_value(const Uint256Case& c) { return uint256_t(c.lo0, c.lo1, c.hi0, c.hi1); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
static const Uint256Case kUint256Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+361 /* 4.6970851655476665e+108 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.0000000000000p-1010 /* 9.1139025244454969e-305 */, 0x1.0000000000000p+256 /* 1.157920892373162e+77 */, 0x1.0000000000000p-754 /* 1.0553178144107943e-227 */, std::numeric_limits<double>::infinity()},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.8a1c1249da8bbp+4 /* 24.631853378764145 */, 0x1.0000000000000p+255 /* 5.7896044618658098e+76 */, 0x1.8a1c1249da8bbp+259 /* 1.4260868822571732e+78 */, 0x1.4c93d67662b1ap+250 /* 2.3504542564617572e+75 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+255 /* 5.7896044618658098e+76 */, 0x1.0000000000000p+255 /* 5.7896044618658098e+76 */, 0x1.0000000000000p+255 /* 5.7896044618658098e+76 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+56 /* 72057594037927936 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+56 /* 72057594037927936 */, 0x1.0000000000000p-56 /* 1.3877787807814457e-17 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0x1.c18709f224424p-28 /* 6.54148600750876e-09 */, 0x1.0000000000000p+255 /* 5.7896044618658098e+76 */, 0x1.c18709f224424p+227 /* 3.7872616576305479e+68 */, 0x1.2393c7fa0b77bp+282 /* 8.8505951938445036e+84 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0123456789ABCDEFULL, 0x1.d6ae3fda26f17p-31 /* 8.5616313628013931e-10 */, 0x1.23456789abcdfp+248 /* 5.1463150772140531e+74 */, 0x1.0bc3d6f4b5717p+218 /* 4.4060852567933507e+65 */, 0x1.3cd73ab1d4536p+278 /* 6.0109047670211329e+83 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.dae0e263dc000p+38 /* 509896857847 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.0e273002320ffp+95 /* 4.1804163736410085e+28 */, 0x1.3a0a3a8cbc756p+17 /* 160788.45741992694 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0x1.0000000000000p-304 /* 3.0681834158110791e-92 */, 0x1.e1e1e1e1e1e1ep+255 /* 1.0898078987041524e+77 */, 0x1.e1e1e1e1e1e1ep-49 /* 3.3437305212240008e-15 */, 0x1.e1e1e1e1e1e1ep+559 /* 3.5519646351261565e+168 */},
    {0x732C649D031EAAC5ULL, 0xE3E0637F9766CF85ULL, 0xFF45DB9D7554F03EULL, 0x0000000000000000ULL, 0x1.0000000000000p+5 /* 32 */, 0x1.fe8bb73aeaa9ep+191 /* 6.2592728615062092e+57 */, 0x1.fe8bb73aeaa9ep+196 /* 2.0029673156819869e+59 */, 0x1.fe8bb73aeaa9ep+186 /* 1.9560227692206904e+56 */},
    {0xED81E5D57C1B327FULL, 0x8F315314D6DD6D0CULL, 0x0000000000000000ULL, 0x84FC76084576BE84ULL, 0x1.844c37b5d0000p+37 /* 208465753786 */, 0x1.09f8ec108aed8p+255 /* 6.015135610237921e+76 */, 0x1.936c75908749ep+292 /* 1.2539497791132592e+88 */, 0x1.5eb46869bc644p+217 /* 2.8854310604957893e+65 */},
    {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x1.0000000000000p+603 /* 3.3196124551047944e+181 */, 0x1.0000000000000p+192 /* 6.2771017353866808e+57 */, 0x1.0000000000000p+795 /* 2.0837545102749545e+239 */, 0x1.0000000000000p-411 /* 1.8909140209225187e-124 */},
    {0x36746DDCDD3ABE10ULL, 0x7FC98E11578BE68AULL, 0x0000000000000005ULL, 0x402264F696D51ADBULL, 0x1.0000000000000p+3 /* 8 */, 0x1.008993da5b547p+254 /* 2.9008791932197585e+76 */, 0x1.008993da5b547p+257 /* 2.3207033545758068e+77 */, 0x1.008993da5b547p+251 /* 3.6260989915246981e+75 */},
    {0x0000000000000000ULL, 0x0000000000000001ULL, 0x000000001DF41C0BULL, 0x0000000080B1F753ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0163eea600000p+223 /* 1.3553184344933159e+67 */, 0x1.0163eea600000p+223 /* 1.3553184344933159e+67 */, 0x1.0163eea600000p+223 /* 1.3553184344933159e+67 */},
    {0x00000000000099AEULL, 0x0000000000000000ULL, 0x5642CC89A6496EC9ULL, 0x1BC8AA1862202EABULL, 0x1.464b64f53c587p+12 /* 5220.7121479375119 */, 0x1.bc8aa1862202fp+252 /* 1.2566990278955388e+76 */, 0x1.1b4dbacf11156p+265 /* 6.5608638812355015e+79 */, 0x1.5cc5d27900a2bp+240 /* 2.4071410035353293e+72 */},
};

constexpr std::size_t kUint256CaseCount = sizeof(kUint256Cases) / sizeof(kUint256Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kUint256UlpSlack = 2.0;

TEST(BigIntFloatUint256, ToDouble)
{
    for (std::size_t i = 0; i < kUint256CaseCount; ++i)
    {
        const Uint256Case& c = kUint256Cases[i];
        const uint256_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kUint256UlpSlack,
                           std::string("uint256_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatUint256, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kUint256CaseCount; ++i)
    {
        const Uint256Case& c = kUint256Cases[i];
        const uint256_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kUint256UlpSlack,
                           std::string("uint256_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatUint256, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kUint256CaseCount; ++i)
    {
        const Uint256Case& c = kUint256Cases[i];
        const uint256_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kUint256UlpSlack,
                           std::string("uint256_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// uint512_t
// ---------------------------------------------------------------------------

struct Uint512Case
{
    uint64_t lo0;
    uint64_t lo1;
    uint64_t lo2;
    uint64_t lo3;
    uint64_t lo4;
    uint64_t lo5;
    uint64_t lo6;
    uint64_t lo7;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static uint512_t make_value(const Uint512Case& c) { return uint512_t(c.lo0, c.lo1, c.lo2, c.lo3, c.lo4, c.lo5, c.lo6, c.lo7); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
static const Uint512Case kUint512Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.188f7caba4b88p+3 /* 8.7675155022059386 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.0000000000000p+114 /* 2.0769187434139311e+34 */, 0x1.0000000000000p+512 /* 1.3407807929942597e+154 */, 0x1.0000000000000p+626 /* 2.7846927597791719e+188 */, 0x1.0000000000000p+398 /* 6.4556246952172715e+119 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p+36 /* 68719476736 */, 0x1.0000000000000p+511 /* 6.7039039649712985e+153 */, 0x1.0000000000000p+547 /* 4.6068877256122331e+164 */, 0x1.0000000000000p+475 /* 9.7554642197374757e+142 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+511 /* 6.7039039649712985e+153 */, 0x1.0000000000000p+511 /* 6.7039039649712985e+153 */, 0x1.0000000000000p+511 /* 6.7039039649712985e+153 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.1192c91b92000p+39 /* 587494362569 */, 0x1.0000000000000p+0 /* 1 */, 0x1.1192c91b92000p+39 /* 587494362569 */, 0x1.df1c669c358d6p-40 /* 1.7021439927137207e-12 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.13bad773fb6e5p+15 /* 35293.420806748596 */, 0x1.0000000000000p+0 /* 1 */, 0x1.13bad773fb6e5p+15 /* 35293.420806748596 */, 0x1.db5d248a92ff5p-16 /* 2.8333892752294106e-05 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0x1.0000000000000p-199 /* 1.2446030555722283e-60 */, 0x1.0000000000000p+511 /* 6.7039039649712985e+153 */, 0x1.0000000000000p+312 /* 8.343699359066055e+93 */, 0x1.0000000000000p+710 /* 5.3863791631855345e+213 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0123456789ABCDEFULL, 0x1.d51197d4b3489p+25 /* 61481775.661721297 */, 0x1.23456789abcdfp+504 /* 5.9590257466411543e+151 */, 0x1.0ad9158584b66p+530 /* 3.6637148411741271e+159 */, 0x1.3dedf71c1085cp+478 /* 9.6923448981504576e+143 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.7f098396a7c95p+21 /* 3137840.4485622146 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.b3cfa90f896d9p+77 /* 2.5725750977227178e+23 */, 0x1.85567392fe4e8p+34 /* 26128010827.973541 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0x1.0000000000000p+142 /* 5.5751862996326558e+42 */, 0x1.e1e1e1e1e1e1ep+511 /* 1.2619113345828327e+154 */, 0x1.e1e1e1e1e1e1ep+653 /* 7.035390783917369e+196 */, 0x1.e1e1e1e1e1e1ep+369 /* 2.2634424515392049e+111 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+41 /* 2199023255552 */, 0x1.0000000000000p+192 /* 6.2771017353866808e+57 */, 0x1.0000000000000p+233 /* 1.3803492693581128e+70 */, 0x1.0000000000000p+151 /* 2.8544953854119198e+45 */},
    {0x0E093D8DF4780D41ULL, 0x552CBCC2AF5520DBULL, 0x2513345C0C71B86CULL, 0x0EF4F715028520D1ULL, 0xE4367BC02AAE2902ULL, 0x1393EB75175A8AA0ULL, 0x3F70EA52F7B38C55ULL, 0x71FA9ACFA79B3ADEULL, 0x1.0000000000000p+122 /* 5.3169119831396635e+36 */, 0x1.c7ea6b3e9e6cfp+510 /* 5.9695606673852469e+153 */, 0x1.c7ea6b3e9e6cfp+632 /* 3.1739628646499826e+190 */, 0x1.c7ea6b3e9e6cfp+388 /* 1.1227495746243651e+117 */},
    {0xFE33EC76D039887DULL, 0x0000000000000007ULL, 0x000000000000000BULL, 0x0000000000000001ULL, 0x000000000000B626ULL, 0x0000000000008DEBULL, 0x000000005F303869ULL, 0x37E3130143C62C67ULL, 0x1.942218e8ec000p+39 /* 867869422710 */, 0x1.bf18980a1e316p+509 /* 2.9270401523312885e+153 */, 0x1.60e72e5aa0d9bp+549 /* 2.5402886472527459e+165 */, 0x1.1b370c25cdfa2p+470 /* 3.3726734411166873e+141 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x1.4000000000000p+1 /* 2.5 */, 0x1.0000000000000p+448 /* 7.2683872429560689e+134 */, 0x1.4000000000000p+449 /* 1.8170968107390172e+135 */, 0x1.999999999999ap+446 /* 2.9073548971824277e+134 */},
    {0xEB429D4B8ADBE2FBULL, 0xF442B4E3557EEFC2ULL, 0xAEA338E64E976CA7ULL, 0xE3BE07D5B00F813BULL, 0x5733454AA9AFEFACULL, 0xB8DE4334F813E3DAULL, 0x48CB3CE81B9716EFULL, 0x8BE5360A85C55413ULL, 0x1.0000000000000p+492 /* 1.2786682062094304e+148 */, 0x1.17ca6c150b8abp+511 /* 7.3269143029372243e+153 */, 0x1.17ca6c150b8abp+1003 /* 9.3686923687869599e+301 */, 0x1.17ca6c150b8abp+19 /* 573011.3775689801 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x6F2C3E25704EA4E0ULL, 0x895ED632B37BC0BFULL, 0x0000000000000006ULL, 0x000000007A873259ULL, 0x000000002DB3A907ULL, 0x1.0000000000000p-4 /* 0.0625 */, 0x1.6d9d483800000p+477 /* 5.5730281777041521e+143 */, 0x1.6d9d483800000p+473 /* 3.4831426110650951e+142 */, 0x1.6d9d483800000p+481 /* 8.9168450843266433e+144 */},
};

constexpr std::size_t kUint512CaseCount = sizeof(kUint512Cases) / sizeof(kUint512Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kUint512UlpSlack = 2.0;

TEST(BigIntFloatUint512, ToDouble)
{
    for (std::size_t i = 0; i < kUint512CaseCount; ++i)
    {
        const Uint512Case& c = kUint512Cases[i];
        const uint512_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kUint512UlpSlack,
                           std::string("uint512_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatUint512, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kUint512CaseCount; ++i)
    {
        const Uint512Case& c = kUint512Cases[i];
        const uint512_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kUint512UlpSlack,
                           std::string("uint512_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatUint512, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kUint512CaseCount; ++i)
    {
        const Uint512Case& c = kUint512Cases[i];
        const uint512_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kUint512UlpSlack,
                           std::string("uint512_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// ext_sgn_int128_t
// ---------------------------------------------------------------------------

struct ExtSgnInt128Case
{
    uint64_t lo;
    uint64_t hi;
    bool neg;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static ext_sgn_int128_t make_value(const ExtSgnInt128Case& c) { return ext_sgn_int128_t(c.lo, c.hi, c.neg); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
// The limbs are the unsigned magnitude; `neg` selects the sign.
static const ExtSgnInt128Case kExtSgnInt128Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.dcd6500000000p+29 /* 1000000000 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, false, 0x1.d24d011ce7727p-31 /* 8.4819621084826772e-10 */, 0x1.0000000000000p+128 /* 3.4028236692093846e+38 */, 0x1.d24d011ce7727p+97 /* 2.8862621424081992e+29 */, 0x1.1916c52e128abp+158 /* 4.0118354994845756e+47 */},
    {0x0000000000000000ULL, 0x8000000000000000ULL, false, 0x1.afb28065d25e7p-18 /* 6.4327906324572779e-06 */, 0x1.0000000000000p+127 /* 1.7014118346046923e+38 */, 0x1.afb28065d25e7p+109 /* 1.0944826111597016e+33 */, 0x1.2f9ec39c1f2f7p+144 /* 2.6449047261386864e+43 */},
    {0x0000000000000000ULL, 0x8000000000000000ULL, true, 0x1.0000000000000p-723 /* 2.2662777498902796e-218 */, -0x1.0000000000000p+127 /* -1.7014118346046923e+38 */, -0x1.0000000000000p-596 /* -3.8558717841646146e-180 */, -0x1.0000000000000p+850 /* -7.5075168288047002e+255 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, false, 0x1.2c4cfb0f4c000p+38 /* 322445427667 */, 0x1.0000000000000p+0 /* 1 */, 0x1.2c4cfb0f4c000p+38 /* 322445427667 */, 0x1.b4781b8f56e0cp-39 /* 3.1012999850403612e-12 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, true, 0x1.0000000000000p-114 /* 4.8148248609680896e-35 */, -0x1.0000000000000p+0 /* -1 */, -0x1.0000000000000p-114 /* -4.8148248609680896e-35 */, -0x1.0000000000000p+114 /* -2.0769187434139311e+34 */},
    {0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, false, 0x1.0000000000000p-53 /* 1.1102230246251565e-16 */, 0x1.0000000000000p+127 /* 1.7014118346046923e+38 */, 0x1.0000000000000p+74 /* 1.8889465931478581e+22 */, 0x1.0000000000000p+180 /* 1.5324955408658889e+54 */},
    {0x0000000000000000ULL, 0x0123456789ABCDEFULL, false, 0x1.0c6f7a0b5ed8dp-20 /* 9.9999999999999995e-07 */, 0x1.23456789abcdfp+120 /* 1.5123660752041709e+36 */, 0x1.316b7e5807ca5p+100 /* 1.5123660752041709e+30 */, 0x1.15c71c71c71c7p+140 /* 1.5123660752041709e+42 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, false, 0x1.97d316b91f1ecp+15 /* 52201.54438111544 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.d00383aace131p+71 /* 4.2797712420036773e+21 */, 0x1.6dac863d4a7d7p+40 /* 1570557541706.49 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, false, 0x1.0000000000000p+0 /* 1 */, 0x1.e1e1e1e1e1e1ep+127 /* 3.2026575710205973e+38 */, 0x1.e1e1e1e1e1e1ep+127 /* 3.2026575710205973e+38 */, 0x1.e1e1e1e1e1e1ep+127 /* 3.2026575710205973e+38 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, true, 0x1.301f96e1e3ea9p+12 /* 4865.9743365195254 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, true, 0x1.0000000000000p+0 /* 1 */, -0x1.0000000000000p+128 /* -3.4028236692093846e+38 */, -0x1.0000000000000p+128 /* -3.4028236692093846e+38 */, -0x1.0000000000000p+128 /* -3.4028236692093846e+38 */},
    {0x093FB669F21B007EULL, 0x0000000000000001ULL, true, 0x1.0000000000000p+0 /* 1 */, -0x1.093fb669f21b0p+64 /* -1.9113195909733351e+19 */, -0x1.093fb669f21b0p+64 /* -1.9113195909733351e+19 */, -0x1.093fb669f21b0p+64 /* -1.9113195909733351e+19 */},
    {0x3697D6B65FD55437ULL, 0x00000000095FD958ULL, false, 0x1.0000000000000p-1 /* 0.5 */, 0x1.2bfb2b06d2fadp+91 /* 2.9012394220296052e+27 */, 0x1.2bfb2b06d2fadp+90 /* 1.4506197110148026e+27 */, 0x1.2bfb2b06d2fadp+92 /* 5.8024788440592105e+27 */},
    {0xFFE007FD11EB1A59ULL, 0x7DBA9126FA5A68AAULL, false, 0x1.0000000000000p-195 /* 1.9913648889155653e-59 */, 0x1.f6ea449be969ap+126 /* 1.6712221072634521e+38 */, 0x1.f6ea449be969ap-69 /* 3.3280130259839214e-21 */, 0x1.f6ea449be969ap+321 /* 8.3923449517760008e+96 */},
    {0xA5C693434AC35628ULL, 0x8C01AC62E785F704ULL, false, 0x1.0624dd2f1a9fcp-10 /* 0.001 */, 0x1.180358c5cf0bfp+127 /* 1.8610060811721151e+38 */, 0x1.1ebbbf4157167p+117 /* 1.8610060811721149e+35 */, 0x1.117344b12c31ap+137 /* 1.8610060811721149e+41 */},
};

constexpr std::size_t kExtSgnInt128CaseCount = sizeof(kExtSgnInt128Cases) / sizeof(kExtSgnInt128Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kExtSgnInt128UlpSlack = 2.0;

TEST(BigIntFloatExtSgnInt128, ToDouble)
{
    for (std::size_t i = 0; i < kExtSgnInt128CaseCount; ++i)
    {
        const ExtSgnInt128Case& c = kExtSgnInt128Cases[i];
        const ext_sgn_int128_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kExtSgnInt128UlpSlack,
                           std::string("ext_sgn_int128_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatExtSgnInt128, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kExtSgnInt128CaseCount; ++i)
    {
        const ExtSgnInt128Case& c = kExtSgnInt128Cases[i];
        const ext_sgn_int128_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kExtSgnInt128UlpSlack,
                           std::string("ext_sgn_int128_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatExtSgnInt128, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kExtSgnInt128CaseCount; ++i)
    {
        const ExtSgnInt128Case& c = kExtSgnInt128Cases[i];
        const ext_sgn_int128_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kExtSgnInt128UlpSlack,
                           std::string("ext_sgn_int128_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// ext_sgn_int192_t
// ---------------------------------------------------------------------------

struct ExtSgnInt192Case
{
    uint64_t limb0;
    uint64_t limb1;
    uint64_t limb2;
    bool neg;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static ext_sgn_int192_t make_value(const ExtSgnInt192Case& c) { return ext_sgn_int192_t(c.limb0, c.limb1, c.limb2, c.neg); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
// The limbs are the unsigned magnitude; `neg` selects the sign.
static const ExtSgnInt192Case kExtSgnInt192Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.0000000000000p+0 /* 1 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, false, 0x1.0000000000000p+17 /* 131072 */, 0x1.0000000000000p+192 /* 6.2771017353866808e+57 */, 0x1.0000000000000p+209 /* 8.2275227866060302e+62 */, 0x1.0000000000000p+175 /* 4.7890485652059027e+52 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, false, 0x1.dcd6500000000p+29 /* 1000000000 */, 0x1.0000000000000p+191 /* 3.1385508676933404e+57 */, 0x1.dcd6500000000p+220 /* 3.1385508676933404e+66 */, 0x1.12e0be826d695p+161 /* 3.1385508676933406e+48 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, true, 0x1.0000000000000p-54 /* 5.5511151231257827e-17 */, -0x1.0000000000000p+191 /* -3.1385508676933404e+57 */, -0x1.0000000000000p+137 /* -1.7422457186352049e+41 */, -0x1.0000000000000p+245 /* -5.6539106072908299e+73 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.0000000000000p+21 /* 2097152 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+21 /* 2097152 */, 0x1.0000000000000p-21 /* 4.76837158203125e-07 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, true, 0x1.51b3c0c22c000p+38 /* 362604933259 */, -0x1.0000000000000p+0 /* -1 */, -0x1.51b3c0c22c000p+38 /* -362604933259 */, -0x1.842105d23fb9ep-39 /* -2.7578223798894761e-12 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, false, 0x1.0c6f7a0b5ed8dp-20 /* 9.9999999999999995e-07 */, 0x1.0000000000000p+191 /* 3.1385508676933404e+57 */, 0x1.0c6f7a0b5ed8dp+171 /* 3.1385508676933402e+51 */, 0x1.e848000000000p+210 /* 3.1385508676933404e+63 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0123456789ABCDEFULL, false, 0x1.0000000000000p-572 /* 6.4690793791235119e-173 */, 0x1.23456789abcdfp+184 /* 2.7898229935051914e+55 */, 0x1.23456789abcdfp-388 /* 1.8047586398689061e-117 */, 0x1.23456789abcdfp+756 /* 4.3125502563908581e+227 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.0000000000000p+0 /* 1 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, false, 0x1.0000000000000p-129 /* 1.4693679385278594e-39 */, 0x1.e1e1e1e1e1e1ep+191 /* 5.907860456834523e+57 */, 0x1.e1e1e1e1e1e1ep+62 /* 8.6808207405692006e+18 */, 0x1.e1e1e1e1e1e1ep+320 /* 4.0206814793805366e+96 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, true, 0x1.7dc285a90a000p+39 /* 819823039621 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, true, 0x1.0000000000000p+5 /* 32 */, -0x1.0000000000000p+192 /* -6.2771017353866808e+57 */, -0x1.0000000000000p+197 /* -2.0086725553237378e+59 */, -0x1.0000000000000p+187 /* -1.9615942923083377e+56 */},
    {0xCEC8D94485C30CD6ULL, 0x85CE1A998CD83AC9ULL, 0x5CAE71FEE957EBF9ULL, true, 0x1.a25ccf8f941b7p+29 /* 877369841.94731796 */, -0x1.72b9c7fba55fbp+190 /* -2.2725419760344411e+57 */, -0x1.2eecdff073b60p+220 /* -1.9938597943319832e+66 */, -0x1.c5b37dec0f85ap+160 /* -2.5901756219367486e+48 */},
    {0x0000000000000001ULL, 0x000000000000E86CULL, 0x00000000A6776622ULL, true, 0x1.0000000000000p+1007 /* 1.3715310171984222e+303 */, -0x1.4ceecc4400000p+159 /* -9.5035515365814802e+47 */, -std::numeric_limits<double>::infinity(), -0x1.4ceecc4400000p-848 /* -6.9291553872358267e-256 */},
    {0xCE91F7CE19FEAA8BULL, 0x0000000000003E94ULL, 0x0000000024D72444ULL, false, 0x1.0000000000000p+0 /* 1 */, 0x1.26b9222000000p+157 /* 2.103214871488368e+47 */, 0x1.26b9222000000p+157 /* 2.103214871488368e+47 */, 0x1.26b9222000000p+157 /* 2.103214871488368e+47 */},
    {0x0000000000000000ULL, 0xBA08DC5712900EB8ULL, 0x0000000000000000ULL, false, 0x1.0000000000000p+59 /* 5.7646075230342349e+17 */, 0x1.7411b8ae25202p+127 /* 2.4728241461957046e+38 */, 0x1.7411b8ae25202p+186 /* 1.4254860676300468e+56 */, 0x1.7411b8ae25202p+68 /* 4.289666098367993e+20 */},
};

constexpr std::size_t kExtSgnInt192CaseCount = sizeof(kExtSgnInt192Cases) / sizeof(kExtSgnInt192Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kExtSgnInt192UlpSlack = 2.0;

TEST(BigIntFloatExtSgnInt192, ToDouble)
{
    for (std::size_t i = 0; i < kExtSgnInt192CaseCount; ++i)
    {
        const ExtSgnInt192Case& c = kExtSgnInt192Cases[i];
        const ext_sgn_int192_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kExtSgnInt192UlpSlack,
                           std::string("ext_sgn_int192_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatExtSgnInt192, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kExtSgnInt192CaseCount; ++i)
    {
        const ExtSgnInt192Case& c = kExtSgnInt192Cases[i];
        const ext_sgn_int192_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kExtSgnInt192UlpSlack,
                           std::string("ext_sgn_int192_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatExtSgnInt192, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kExtSgnInt192CaseCount; ++i)
    {
        const ExtSgnInt192Case& c = kExtSgnInt192Cases[i];
        const ext_sgn_int192_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kExtSgnInt192UlpSlack,
                           std::string("ext_sgn_int192_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// ext_sgn_int256_t
// ---------------------------------------------------------------------------

struct ExtSgnInt256Case
{
    uint64_t lo0;
    uint64_t lo1;
    uint64_t hi0;
    uint64_t hi1;
    bool neg;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static ext_sgn_int256_t make_value(const ExtSgnInt256Case& c) { return ext_sgn_int256_t(c.lo0, c.lo1, c.hi0, c.hi1, c.neg); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
// The limbs are the unsigned magnitude; `neg` selects the sign.
static const ExtSgnInt256Case kExtSgnInt256Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.4342bd219b1bep-18 /* 4.8169597747057705e-06 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, false, 0x1.0000000000000p-2 /* 0.25 */, 0x1.0000000000000p+256 /* 1.157920892373162e+77 */, 0x1.0000000000000p+254 /* 2.8948022309329049e+76 */, 0x1.0000000000000p+258 /* 4.6316835694926478e+77 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, false, 0x1.d2052c6e68000p+37 /* 250192694733 */, 0x1.0000000000000p+255 /* 5.7896044618658098e+76 */, 0x1.d2052c6e68000p+292 /* 1.4485167417524073e+88 */, 0x1.194218a9ac614p+217 /* 2.3140581574711224e+65 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, true, 0x1.0000000000000p+23 /* 8388608 */, -0x1.0000000000000p+255 /* -5.7896044618658098e+76 */, -0x1.0000000000000p+278 /* -4.8566722305643227e+83 */, -0x1.0000000000000p+232 /* -6.9017463467905638e+69 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.0000000000000p-50 /* 8.8817841970012523e-16 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p-50 /* 8.8817841970012523e-16 */, 0x1.0000000000000p+50 /* 1125899906842624 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, true, 0x1.0000000000000p-200 /* 6.2230152778611417e-61 */, -0x1.0000000000000p+0 /* -1 */, -0x1.0000000000000p-200 /* -6.2230152778611417e-61 */, -0x1.0000000000000p+200 /* -1.6069380442589903e+60 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, false, 0x1.627d6caf6ab52p+29 /* 743419285.92710328 */, 0x1.0000000000000p+255 /* 5.7896044618658098e+76 */, 0x1.627d6caf6ab52p+284 /* 4.3041036148406514e+85 */, 0x1.71bf86df72390p+225 /* 7.7878050401204077e+67 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0123456789ABCDEFULL, false, 0x1.20a7bf17eabddp-6 /* 0.017618118872331102 */, 0x1.23456789abcdfp+248 /* 5.1463150772140531e+74 */, 0x1.486cf02d6aa83p+242 /* 9.0668390784827003e+72 */, 0x1.0251e679d9e52p+254 /* 2.9210355058372527e+76 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.0dc3ff58e9017p+27 /* 141434874.77844307 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.32eeee30d27edp+83 /* 1.1595613058378211e+25 */, 0x1.14687a0ded15cp+29 /* 579669825.74076414 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, false, 0x1.0000000000000p+63 /* 9.2233720368547758e+18 */, 0x1.e1e1e1e1e1e1ep+255 /* 1.0898078987041524e+77 */, 0x1.e1e1e1e1e1e1ep+318 /* 1.0051703698451341e+96 */, 0x1.e1e1e1e1e1e1ep+192 /* 1.1815720913669046e+58 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, true, 0x1.99ed14d9dcd60p-1 /* 0.8006369129129034 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, true, 0x1.0000000000000p+78 /* 3.0223145490365729e+23 */, -0x1.0000000000000p+256 /* -1.157920892373162e+77 */, -0x1.0000000000000p+334 /* -3.4996011596528191e+100 */, -0x1.0000000000000p+178 /* -3.8312388521647221e+53 */},
    {0x0000000000000000ULL, 0x11E5F74737F2EC0BULL, 0x00000000B1267CABULL, 0x0000000000000000ULL, false, 0x1.afd2bf3d00000p+38 /* 463666663232 */, 0x1.624cf95623cbfp+159 /* 1.011349654888895e+48 */, 0x1.2ad1a1cc34a63p+198 /* 4.6892911984316868e+59 */, 0x1.a415749271144p+120 /* 2.1811998469746716e+36 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0xD4D315699FC1FE7DULL, 0x9A7CE5537C855414ULL, true, 0x1.0000000000000p+453 /* 2.325883917745942e+136 */, -0x1.34f9caa6f90abp+255 /* -6.987685046855811e+76 */, -0x1.34f9caa6f90abp+708 /* -1.625254427275573e+213 */, -0x1.34f9caa6f90abp-198 /* -3.0043137551024938e-60 */},
    {0x0C9457921B4698DEULL, 0x10A4EB38E608AF4EULL, 0x660AEF6983393E71ULL, 0xD9C1C0DDC12E0B34ULL, false, 0x1.999999999999ap-4 /* 0.10000000000000001 */, 0x1.b38381bb825c1p+255 /* 9.849422073986204e+76 */, 0x1.5c69349601e35p+252 /* 9.8494220739862056e+75 */, 0x1.1032311531799p+259 /* 9.8494220739862048e+77 */},
    {0x0000000000000000ULL, 0xFDA989A88744C78AULL, 0x0000000000000000ULL, 0x478AFCAD9F000BB6ULL, true, 0x1.0000000000000p-58 /* 3.4694469519536142e-18 */, -0x1.1e2bf2b67c003p+254 /* -3.2359781065246169e+76 */, -0x1.1e2bf2b67c003p+196 /* -1.122705437827046e+59 */, -0x1.1e2bf2b67c003p+312 /* -9.3270718686229426e+93 */},
};

constexpr std::size_t kExtSgnInt256CaseCount = sizeof(kExtSgnInt256Cases) / sizeof(kExtSgnInt256Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kExtSgnInt256UlpSlack = 2.0;

TEST(BigIntFloatExtSgnInt256, ToDouble)
{
    for (std::size_t i = 0; i < kExtSgnInt256CaseCount; ++i)
    {
        const ExtSgnInt256Case& c = kExtSgnInt256Cases[i];
        const ext_sgn_int256_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kExtSgnInt256UlpSlack,
                           std::string("ext_sgn_int256_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatExtSgnInt256, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kExtSgnInt256CaseCount; ++i)
    {
        const ExtSgnInt256Case& c = kExtSgnInt256Cases[i];
        const ext_sgn_int256_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kExtSgnInt256UlpSlack,
                           std::string("ext_sgn_int256_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatExtSgnInt256, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kExtSgnInt256CaseCount; ++i)
    {
        const ExtSgnInt256Case& c = kExtSgnInt256Cases[i];
        const ext_sgn_int256_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kExtSgnInt256UlpSlack,
                           std::string("ext_sgn_int256_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// ext_sgn_int512_t
// ---------------------------------------------------------------------------

struct ExtSgnInt512Case
{
    uint64_t lo0;
    uint64_t lo1;
    uint64_t lo2;
    uint64_t lo3;
    uint64_t lo4;
    uint64_t lo5;
    uint64_t lo6;
    uint64_t lo7;
    bool neg;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static ext_sgn_int512_t make_value(const ExtSgnInt512Case& c) { return ext_sgn_int512_t(uint512_t(c.lo0, c.lo1, c.lo2, c.lo3, c.lo4, c.lo5, c.lo6, c.lo7), c.neg); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
// The limbs are the unsigned magnitude; `neg` selects the sign.
static const ExtSgnInt512Case kExtSgnInt512Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.7154549d37b76p-21 /* 6.8792964538407129e-07 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, false, 0x1.dcd6500000000p+29 /* 1000000000 */, 0x1.0000000000000p+512 /* 1.3407807929942597e+154 */, 0x1.dcd6500000000p+541 /* 1.3407807929942597e+163 */, 0x1.12e0be826d695p+482 /* 1.3407807929942598e+145 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, false, 0x1.0c529fd976000p+39 /* 576218721467 */, 0x1.0000000000000p+511 /* 6.7039039649712985e+153 */, 0x1.0c529fd976000p+550 /* 3.8629149715333136e+165 */, 0x1.e87c810dbdd8bp+471 /* 1.1634304327883991e+142 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, true, 0x1.0000000000000p-41 /* 4.5474735088646412e-13 */, -0x1.0000000000000p+511 /* -6.7039039649712985e+153 */, -0x1.0000000000000p+470 /* -3.0485825686679612e+141 */, -0x1.0000000000000p+552 /* -1.4742040721959146e+166 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, true, 0x1.0000000000000p-31 /* 4.6566128730773926e-10 */, -0x1.0000000000000p+0 /* -1 */, -0x1.0000000000000p-31 /* -4.6566128730773926e-10 */, -0x1.0000000000000p+31 /* -2147483648 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, false, 0x1.84ac7bd25c000p+39 /* 834670553390 */, 0x1.0000000000000p+511 /* 6.7039039649712985e+153 */, 0x1.84ac7bd25c000p+550 /* 5.5955512323160089e+165 */, 0x1.513a94cdb4284p+471 /* 8.031796422844329e+141 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0123456789ABCDEFULL, false, 0x1.0000000000000p-41 /* 4.5474735088646412e-13 */, 0x1.23456789abcdfp+504 /* 5.9590257466411543e+151 */, 0x1.23456789abcdfp+463 /* 2.7098511721492988e+139 */, 0x1.23456789abcdfp+545 /* 1.3104036197297019e+164 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, false, 0x1.0000000000000p-827 /* 1.1173611982879273e-249 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.23456789abcdfp-771 /* 9.1607449167603674e-233 */, 0x1.23456789abcdfp+883 /* 7.3374240435509062e+265 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, false, 0x1.2fca5154a8000p+38 /* 326192354602 */, 0x1.e1e1e1e1e1e1ep+511 /* 1.2619113345828327e+154 */, 0x1.1deb97d734b4bp+550 /* 4.116258295265264e+165 */, 0x1.961365dd6de6ap+473 /* 3.8686110105877248e+142 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, true, 0x1.604b738ca8a10p-15 /* 4.1996804701644711e-05 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, true, 0x1.5607bc895afcdp+14 /* 21889.934117719346 */, -0x1.0000000000000p+512 /* -1.3407807929942597e+154 */, -0x1.5607bc895afcdp+526 /* -2.9349603224947845e+158 */, -0x1.7f37b4955d085p+497 /* -6.1251020025178228e+149 */},
    {0xD7DC278DB6E386F2ULL, 0x2A1A9588AFD00FEAULL, 0x0000000000000000ULL, 0xF05B153BC6C76E41ULL, 0xAB63BF15F8DDDE42ULL, 0x8D06BEE53BC1C6E5ULL, 0x0000000000000000ULL, 0x0D4127BDF8DB6F3DULL, true, 0x1.a265e5f5deb49p+2 /* 6.5374693776200923 */, -0x1.a824f7bf1b6dep+507 /* -6.9419515636962438e+152 */, -0x1.5a9a980e1b94cp+510 /* -4.538279576858611e+153 */, -0x1.038415c061ed8p+505 /* -1.0618713699005349e+152 */},
    {0x0000000000000000ULL, 0x0000000086200820ULL, 0x8ED43FB7EAC48B84ULL, 0x000000000000BDF0ULL, 0x0000000000000004ULL, 0x0000000000000002ULL, 0x000000006CACB0B3ULL, 0x0000000000005578ULL, false, 0x1.0000000000000p-51 /* 4.4408920985006262e-16 */, 0x1.55e000000001bp+462 /* 1.590323128758795e+139 */, 0x1.55e000000001bp+411 /* 7.0624534165677267e+123 */, 0x1.55e000000001bp+513 /* 3.5810893250383953e+154 */},
    {0x5B21512D71428C72ULL, 0xB9321CAD09011562ULL, 0x1E65072901FBE69EULL, 0x4F9A6FFB03849246ULL, 0x3F0893DF1DB21993ULL, 0xDD78E7ABCB6E6C25ULL, 0x11700A74ACB6501CULL, 0xCEC63E55F81CDED3ULL, true, 0x1.d99999999999ap+1 /* 3.7000000000000002 */, -0x1.9d8c7cabf039cp+511 /* -1.0829653469165585e+154 */, -0x1.7e8859b8a49bdp+513 /* -4.0069717835912664e+154 */, -0x1.bf146b1abe839p+509 /* -2.9269333700447524e+153 */},
    {0x0000000000000000ULL, 0x0000000000000001ULL, 0x00000000000017EFULL, 0x0000000000000001ULL, 0x87778A3932EE7CC8ULL, 0x0000000000000001ULL, 0x0000000000000001ULL, 0x0000000000000009ULL, true, 0x1.12e0be826d695p-30 /* 1.0000000000000001e-09 */, -0x1.2000000000000p+451 /* -6.541548518660462e+135 */, -0x1.353cd652bb168p+421 /* -6.5415485186604629e+126 */, -0x1.0c388d0000000p+481 /* -6.541548518660462e+144 */},
};

constexpr std::size_t kExtSgnInt512CaseCount = sizeof(kExtSgnInt512Cases) / sizeof(kExtSgnInt512Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kExtSgnInt512UlpSlack = 2.0;

TEST(BigIntFloatExtSgnInt512, ToDouble)
{
    for (std::size_t i = 0; i < kExtSgnInt512CaseCount; ++i)
    {
        const ExtSgnInt512Case& c = kExtSgnInt512Cases[i];
        const ext_sgn_int512_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kExtSgnInt512UlpSlack,
                           std::string("ext_sgn_int512_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatExtSgnInt512, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kExtSgnInt512CaseCount; ++i)
    {
        const ExtSgnInt512Case& c = kExtSgnInt512Cases[i];
        const ext_sgn_int512_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kExtSgnInt512UlpSlack,
                           std::string("ext_sgn_int512_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatExtSgnInt512, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kExtSgnInt512CaseCount; ++i)
    {
        const ExtSgnInt512Case& c = kExtSgnInt512Cases[i];
        const ext_sgn_int512_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kExtSgnInt512UlpSlack,
                           std::string("ext_sgn_int512_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// int128_t
// ---------------------------------------------------------------------------

struct Int128Case
{
    uint64_t lo;
    uint64_t hi;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static int128_t make_value(const Int128Case& c) { return int128_t(c.lo, c.hi); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
// The limbs are the raw two's-complement bit pattern, so the top bit carries
// the sign.
static const Int128Case kInt128Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.93af117060b94p+9 /* 807.36771969530901 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.00b6236f72000p+39 /* 551283701689 */, -0x1.0000000000000p+0 /* -1 */, -0x1.00b6236f72000p+39 /* -551283701689 */, -0x1.fe94bb9607d56p-40 /* -1.81394805784434e-12 */},
    {0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.1b5aaded40000p+36 /* 76062318292 */, -0x1.0000000000000p+127 /* -1.7014118346046923e+38 */, -0x1.1b5aaded40000p+163 /* -1.2941332850947777e+49 */, -0x1.ce92add3bda70p+90 /* -2.2368656028508685e+27 */},
    {0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p-50 /* 8.8817841970012523e-16 */, -0x1.0000000000000p+127 /* -1.7014118346046923e+38 */, -0x1.0000000000000p+77 /* -1.5111572745182865e+23 */, -0x1.0000000000000p+177 /* -1.9156194260823611e+53 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.0c6f7a0b5ed8dp-20 /* 9.9999999999999995e-07 */, -0x1.0000000000000p+0 /* -1 */, -0x1.0c6f7a0b5ed8dp-20 /* -9.9999999999999995e-07 */, -0x1.e848000000000p+19 /* -1000000 */},
    {0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0x1.0000000000000p+370 /* 2.4049076047604052e+111 */, 0x1.0000000000000p+127 /* 1.7014118346046923e+38 */, 0x1.0000000000000p+497 /* 4.0917382598701773e+149 */, 0x1.0000000000000p-243 /* 7.074749280333369e-74 */},
    {0x0000000000000000ULL, 0x0123456789ABCDEFULL, 0x1.0000000000000p+23 /* 8388608 */, 0x1.23456789abcdfp+120 /* 1.5123660752041709e+36 */, 0x1.23456789abcdfp+143 /* 1.268664615738631e+43 */, 0x1.23456789abcdfp+97 /* 1.8028808536579263e+29 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x1.0000000000000p-917 /* 9.0259718793241479e-277 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.23456789abcdfp-861 /* 7.3999908121951906e-260 */, 0x1.23456789abcdfp+973 /* 9.0832910087269026e+292 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0x1.0000000000000p-24 /* 5.9604644775390625e-08 */, -0x1.e1e1e1e1e1e1ep+123 /* -2.0016609818878733e+37 */, -0x1.e1e1e1e1e1e1ep+99 /* -1.1930829178618629e+30 */, -0x1.e1e1e1e1e1e1ep+147 /* -3.3582298651904938e+44 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p-27 /* 7.4505805969238281e-09 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x1.0000000000000p+146 /* 8.9202980794122493e+43 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+146 /* 8.9202980794122493e+43 */, 0x1.0000000000000p-146 /* 1.1210387714598537e-44 */},
    {0x763BE9804603F3DFULL, 0x207EED508DCB86F8ULL, 0x1.0000000000000p+227 /* 2.1567957333720512e+68 */, 0x1.03f76a846e5c3p+125 /* 4.3194338582505194e+37 */, 0x1.03f76a846e5c3p+352 /* 9.3161365160574975e+105 */, 0x1.03f76a846e5c3p-102 /* 2.0027088293137907e-31 */},
    {0x79F7554A8A03B66FULL, 0xFDB48B2A4DFBA10FULL, 0x1.0000000000000p-113 /* 9.6296497219361793e-35 */, -0x1.25ba6ad9022f8p+121 /* -3.050247946166199e+36 */, -0x1.25ba6ad9022f8p+8 /* -293.7281928663574 */, -0x1.25ba6ad9022f8p+234 /* -3.167558565726213e+70 */},
    {0x000000000000675EULL, 0x104538E009F376E6ULL, 0x1.014ef54dd4000p+38 /* 276282823541 */, 0x1.04538e009f377p+124 /* 2.1627069980923461e+37 */, 0x1.05a82c76ffa4bp+162 /* 5.9751879592483352e+48 */, 0x1.0300aaf3349e1p+86 /* 7.8278735187871832e+25 */},
    {0x551AE9F950A26F38ULL, 0xBA333BB629AA40AFULL, 0x1.0000000000000p+288 /* 4.9732323640978664e+86 */, -0x1.1733112759570p+126 /* -9.2779941470571937e+37 */, -0x1.1733112759570p+414 /* -4.6141620766055415e+124 */, -0x1.1733112759570p-162 /* -1.8655862963564144e-49 */},
};

constexpr std::size_t kInt128CaseCount = sizeof(kInt128Cases) / sizeof(kInt128Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kInt128UlpSlack = 2.0;

TEST(BigIntFloatInt128, ToDouble)
{
    for (std::size_t i = 0; i < kInt128CaseCount; ++i)
    {
        const Int128Case& c = kInt128Cases[i];
        const int128_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kInt128UlpSlack,
                           std::string("int128_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatInt128, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kInt128CaseCount; ++i)
    {
        const Int128Case& c = kInt128Cases[i];
        const int128_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kInt128UlpSlack,
                           std::string("int128_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatInt128, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kInt128CaseCount; ++i)
    {
        const Int128Case& c = kInt128Cases[i];
        const int128_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kInt128UlpSlack,
                           std::string("int128_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// int192_t
// ---------------------------------------------------------------------------

struct Int192Case
{
    uint64_t limb0;
    uint64_t limb1;
    uint64_t limb2;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static int192_t make_value(const Int192Case& c) { return int192_t(c.limb0, c.limb1, c.limb2); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
// The limbs are the raw two's-complement bit pattern, so the top bit carries
// the sign.
static const Int192Case kInt192Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+835 /* 2.2911123134779969e+251 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.0000000000000p+327 /* 2.7340634059787649e+98 */, -0x1.0000000000000p+0 /* -1 */, -0x1.0000000000000p+327 /* -2.7340634059787649e+98 */, -0x1.0000000000000p-327 /* -3.6575596521032799e-99 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0b7c8afe30000p+37 /* 143605718982 */, -0x1.0000000000000p+191 /* -3.1385508676933404e+57 */, -0x1.0b7c8afe30000p+228 /* -4.507138539166821e+68 */, -0x1.ea0376322ea93p+153 /* -2.1855333408321549e+46 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p+946 /* 5.9480676339111323e+284 */, -0x1.0000000000000p+191 /* -3.1385508676933404e+57 */, -std::numeric_limits<double>::infinity(), -0x1.0000000000000p-755 /* -5.2765890720539716e-228 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.f1f046bbf6000p+39 /* 1069314956795 */, 0x1.0000000000000p+0 /* 1 */, 0x1.f1f046bbf6000p+39 /* 1069314956795 */, 0x1.073ab049dee48p-40 /* 9.3517816583922656e-13 */},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.4000000000000p+1 /* 2.5 */, -0x1.0000000000000p+0 /* -1 */, -0x1.4000000000000p+1 /* -2.5 */, -0x1.999999999999ap-2 /* -0.40000000000000002 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0x1.0000000000000p+20 /* 1048576 */, 0x1.0000000000000p+191 /* 3.1385508676933404e+57 */, 0x1.0000000000000p+211 /* 3.2910091146424121e+63 */, 0x1.0000000000000p+171 /* 2.9931553532536892e+51 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0123456789ABCDEFULL, 0x1.0000000000000p-791 /* 7.6784476871456305e-239 */, 0x1.23456789abcdfp+184 /* 2.7898229935051914e+55 */, 0x1.23456789abcdfp-607 /* 2.1421509912025637e-183 */, 0x1.23456789abcdfp+975 /* 3.633316403490761e+293 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p-63 /* 1.0842021724855044e-19 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.23456789abcdfp-7 /* 0.0088888888888888889 */, 0x1.23456789abcdfp+119 /* 7.5618303760208547e+35 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0x1.0c6f7a0b5ed8dp-20 /* 9.9999999999999995e-07 */, -0x1.e1e1e1e1e1e1ep+187 /* -3.6924127855215769e+56 */, -0x1.f94a4f24764cdp+167 /* -3.6924127855215767e+50 */, -0x1.cb8f0f0f0f0f1p+207 /* -3.6924127855215769e+62 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.79e994443a18fp-31 /* 6.8741868493348002e-10 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+16 /* 65536 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+16 /* 65536 */, 0x1.0000000000000p-16 /* 1.52587890625e-05 */},
    {0x0000000000000001ULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x1.0000000000000p+8 /* 256 */, 0x1.0000000000000p+128 /* 3.4028236692093846e+38 */, 0x1.0000000000000p+136 /* 8.7112285931760247e+40 */, 0x1.0000000000000p+120 /* 1.3292279957849159e+36 */},
    {0x77AA323C9381752BULL, 0xF42D3D97DC6415A5ULL, 0x2CAED4AD47068E3EULL, 0x1.0000000000000p-845 /* 4.2623947078244298e-255 */, 0x1.6576a56a38347p+189 /* 1.0956223216386313e+57 */, 0x1.6576a56a38347p-656 /* 4.6699747855268173e-198 */, std::numeric_limits<double>::infinity()},
    {0xE02C85F4A653883EULL, 0x10CAD67CF57D5617ULL, 0x7E2619997E9D06D6ULL, 0x1.607ffbe80486fp-18 /* 5.2526583900360215e-06 */, 0x1.f8986665fa742p+190 /* 3.0931602652382526e+57 */, 0x1.5b66e876e2330p+173 /* 1.6247314218929752e+52 */, 0x1.6e753b60542a4p+208 /* 5.8887520100408438e+62 */},
    {0x847303B41730CD0CULL, 0x5A2A6F59AB0EA894ULL, 0x7DB9FB7192CA9AB0ULL, 0x1.f7c3c8538d67fp+0 /* 1.967831154249694 */, 0x1.f6e7edc64b2a7p+190 /* 3.0828046376599639e+57 */, 0x1.eed12832083d4p+191 /* 6.0664390084527161e+57 */, 0x1.ff208d608513cp+189 /* 1.5666001785785292e+57 */},
};

constexpr std::size_t kInt192CaseCount = sizeof(kInt192Cases) / sizeof(kInt192Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kInt192UlpSlack = 2.0;

TEST(BigIntFloatInt192, ToDouble)
{
    for (std::size_t i = 0; i < kInt192CaseCount; ++i)
    {
        const Int192Case& c = kInt192Cases[i];
        const int192_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kInt192UlpSlack,
                           std::string("int192_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatInt192, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kInt192CaseCount; ++i)
    {
        const Int192Case& c = kInt192Cases[i];
        const int192_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kInt192UlpSlack,
                           std::string("int192_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatInt192, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kInt192CaseCount; ++i)
    {
        const Int192Case& c = kInt192Cases[i];
        const int192_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kInt192UlpSlack,
                           std::string("int192_t case ") + std::to_string(i));
    }
}


// ---------------------------------------------------------------------------
// int256_t
// ---------------------------------------------------------------------------

struct Int256Case
{
    uint64_t lo0;
    uint64_t lo1;
    uint64_t hi0;
    uint64_t hi1;
    double resolution;
    double as_double;
    double times_resolution;
    double over_resolution;
};

static int256_t make_value(const Int256Case& c) { return int256_t(c.lo0, c.lo1, c.hi0, c.hi1); }

// as_double / times_resolution / over_resolution are the correctly rounded
// references computed offline with Python integers/Fractions.
// The limbs are the raw two's-complement bit pattern, so the top bit carries
// the sign.
static const Int256Case kInt256Cases[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+778 /* 1.5897785265159259e+234 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.0000000000000p+30 /* 1073741824 */, -0x1.0000000000000p+0 /* -1 */, -0x1.0000000000000p+30 /* -1073741824 */, -0x1.0000000000000p-30 /* -9.3132257461547852e-10 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.7410c1ad80000p+34 /* 24968890038 */, -0x1.0000000000000p+255 /* -5.7896044618658098e+76 */, -0x1.7410c1ad80000p+289 /* -1.4455999717184157e+87 */, -0x1.604837b7f60e9p+220 /* -2.3187272053562038e+66 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x1.0000000000000p-4 /* 0.0625 */, -0x1.0000000000000p+255 /* -5.7896044618658098e+76 */, -0x1.0000000000000p+251 /* -3.6185027886661311e+75 */, -0x1.0000000000000p+259 /* -9.2633671389852956e+77 */},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */, 0x1.0000000000000p+0 /* 1 */},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.0000000000000p-477 /* 2.5626663618343692e-144 */, -0x1.0000000000000p+0 /* -1 */, -0x1.0000000000000p-477 /* -2.5626663618343692e-144 */, -0x1.0000000000000p+477 /* -3.9021856878949903e+143 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0x1.0000000000000p+163 /* 1.1692013098647223e+49 */, 0x1.0000000000000p+255 /* 5.7896044618658098e+76 */, 0x1.0000000000000p+418 /* 6.7692131204121457e+125 */, 0x1.0000000000000p+92 /* 4.9517601571415211e+27 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0123456789ABCDEFULL, 0x1.0000000000000p+0 /* 1 */, 0x1.23456789abcdfp+248 /* 5.1463150772140531e+74 */, 0x1.23456789abcdfp+248 /* 5.1463150772140531e+74 */, 0x1.23456789abcdfp+248 /* 5.1463150772140531e+74 */},
    {0x0123456789ABCDEFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+0 /* 1 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */, 0x1.23456789abcdfp+56 /* 81985529216486896 */},
    {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0x1.0000000000000p-144 /* 4.4841550858394146e-44 */, -0x1.e1e1e1e1e1e1ep+251 /* -6.8112993669009526e+75 */, -0x1.e1e1e1e1e1e1ep+107 /* -3.0542922697263692e+32 */, -0x1.e1e1e1e1e1e1ep+395 /* -1.5189705165217109e+119 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.0000000000000p+112 /* 5.1922968585348276e+33 */, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x1.dcd6500000000p+29 /* 1000000000 */, 0x1.0000000000000p+0 /* 1 */, 0x1.dcd6500000000p+29 /* 1000000000 */, 0x1.12e0be826d695p-30 /* 1.0000000000000001e-09 */},
    {0xFFFFFFFFFFFFFDFCULL, 0xFFFFFFFFFFFF0E08ULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL, 0x1.0000000000000p+3 /* 8 */, -0x1.000000000000fp+128 /* -3.402823669209396e+38 */, -0x1.000000000000fp+131 /* -2.7222589353675168e+39 */, -0x1.000000000000fp+125 /* -4.253529586511745e+37 */},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x1.b8100cc50c000p+39 /* 945027441286 */, 0x1.0000000000000p+192 /* 6.2771017353866808e+57 */, 0x1.b8100cc50c000p+231 /* 5.9320333916843852e+69 */, 0x1.29d9354688b7dp+152 /* 6.6422428187320741e+45 */},
    {0x0000000000000000ULL, 0x000000005C4887D3ULL, 0x000000000000000FULL, 0x0000000000000001ULL, 0x1.ed52a1a53a000p+39 /* 1059402601117 */, 0x1.0000000000000p+192 /* 6.2771017353866808e+57 */, 0x1.ed52a1a53a000p+231 /* 6.6499779059446842e+69 */, 0x1.09b13218f0fe4p+152 /* 5.9251333994916627e+45 */},
    {0x8619103537215248ULL, 0xE3E7824B876ABDC4ULL, 0x39B34A0FFD1CF3EAULL, 0x0000000000000000ULL, 0x1.0000000000000p+43 /* 8796093022208 */, 0x1.cd9a507fe8e7ap+189 /* 1.4148084371876903e+57 */, 0x1.cd9a507fe8e7ap+232 /* 1.2444786622107648e+70 */, 0x1.cd9a507fe8e7ap+146 /* 1.6084509720573013e+44 */},
};

constexpr std::size_t kInt256CaseCount = sizeof(kInt256Cases) / sizeof(kInt256Cases[0]);

// The limb-by-limb accumulation is within one ulp of the correctly rounded
// reference; two ulps is a safety margin.
constexpr double kInt256UlpSlack = 2.0;

TEST(BigIntFloatInt256, ToDouble)
{
    for (std::size_t i = 0; i < kInt256CaseCount; ++i)
    {
        const Int256Case& c = kInt256Cases[i];
        const int256_t value = make_value(c);
        expect_close_double(to_double(value), c.as_double, kInt256UlpSlack,
                           std::string("int256_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatInt256, ToDoubleScaledBy)
{
    for (std::size_t i = 0; i < kInt256CaseCount; ++i)
    {
        const Int256Case& c = kInt256Cases[i];
        const int256_t value = make_value(c);
        expect_close_double(to_double_scaled_by(value, c.resolution), c.times_resolution, kInt256UlpSlack,
                           std::string("int256_t case ") + std::to_string(i));
    }
}

TEST(BigIntFloatInt256, ToDoubleDividedBy)
{
    for (std::size_t i = 0; i < kInt256CaseCount; ++i)
    {
        const Int256Case& c = kInt256Cases[i];
        const int256_t value = make_value(c);
        expect_close_double(to_double_divided_by(value, c.resolution), c.over_resolution, kInt256UlpSlack,
                           std::string("int256_t case ") + std::to_string(i));
    }
}
