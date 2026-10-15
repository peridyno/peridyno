#!/usr/bin/env python3
"""Generate tests/Cuda/Test_Core/Test_BigIntFloat.cpp.

Random and extreme conversion cases are built with Python's arbitrary precision
integers, and the expected doubles are computed exactly (integers / Fractions)
and rounded once to the nearest double.  They are therefore correctly rounded
references that are independent from the C++ implementation.

Usage:
    python tools/gen_test_bigint_float.py
"""

import math
import os
import random
from fractions import Fraction

SEED = 0x5EEDBE1F
ROWS = 16
DBL_MIN = 2.2250738585072014e-308
# The limb-by-limb accumulation stays within one ulp of the correctly rounded
# reference (measured on MSVC, where long double is only 64 bits wide); two ulps
# leave a small margin for other platforms and optimisation levels.
ULP_SLACK = 2.0

# (C++ type, test id, limb field names low -> high, value encoding)
TYPES = [
    ("uint128_t", "Uint128", ("lo", "hi"), "unsigned"),
    ("uint192_t", "Uint192", ("lo", "mi", "hi"), "unsigned"),
    ("uint256_t", "Uint256", ("lo0", "lo1", "hi0", "hi1"), "unsigned"),
    ("uint512_t", "Uint512", tuple("lo%d" % i for i in range(8)), "unsigned"),
    ("ext_sgn_int128_t", "ExtSgnInt128", ("lo", "hi"), "magnitude"),
    ("ext_sgn_int192_t", "ExtSgnInt192", ("limb0", "limb1", "limb2"), "magnitude"),
    ("ext_sgn_int256_t", "ExtSgnInt256", ("lo0", "lo1", "hi0", "hi1"), "magnitude"),
    ("ext_sgn_int512_t", "ExtSgnInt512", tuple("lo%d" % i for i in range(8)), "magnitude"),
    ("int128_t", "Int128", ("lo", "hi"), "twos"),
    ("int192_t", "Int192", ("limb0", "limb1", "limb2"), "twos"),
    ("int256_t", "Int256", ("lo0", "lo1", "hi0", "hi1"), "twos"),
]


def exact_to_double(value):
    """Round an exact integer/Fraction to the nearest double (may overflow)."""
    if value == 0:
        return 0.0
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def unsigned_value(limbs):
    return sum(limb << (64 * i) for i, limb in enumerate(limbs))


def twos_value(limbs):
    """Value of a raw two's-complement bit pattern."""
    magnitude = unsigned_value(limbs)
    top_bit = 64 * len(limbs) - 1
    return magnitude - (1 << (top_bit + 1)) if (magnitude >> top_bit) & 1 else magnitude


def twos_pattern(limbs):
    """Two's-complement bit pattern of the negated magnitude."""
    modulus = 1 << (64 * len(limbs))
    negated = (-unsigned_value(limbs)) % modulus
    return [(negated >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(len(limbs))]


def normalize(limbs, neg, kind):
    """Return (stored limbs, stored negative flag, exact value)."""
    if kind == "unsigned":
        return list(limbs), False, unsigned_value(limbs)
    if kind == "magnitude":
        magnitude = unsigned_value(limbs)
        return list(limbs), neg, (-magnitude if neg else magnitude)
    stored = twos_pattern(limbs) if neg else list(limbs)
    return stored, False, twos_value(stored)


def structural_rows(nlimbs, kind):
    """Extreme / boundary bit patterns that must always be covered."""
    full = [0xFFFFFFFFFFFFFFFF] * nlimbs
    zero = [0] * nlimbs
    only_top = [0] * nlimbs
    only_top[-1] = 1 << 63

    rows = [
        (zero, False),                       # 0
        (full, False),                       # 2^bits - 1
        (only_top, False),                   # 2^(bits-1)
        (only_top, True),                    # -2^(bits-1)
        ([1] + [0] * (nlimbs - 1), False),   # 1
        ([1] + [0] * (nlimbs - 1), True),    # -1
        ([0] * (nlimbs - 1) + [0x7FFFFFFFFFFFFFFF], False),
        ([0] * (nlimbs - 1) + [0x0123456789ABCDEF], False),
        ([0x0123456789ABCDEF] + [0] * (nlimbs - 1), False),
        ([0xF0F0F0F0F0F0F0F0] * nlimbs, False),
    ]
    if kind != "unsigned":
        rows.append((zero, True))            # signed zero
        rows.append((full, True))            # -(2^bits - 1)
    return rows


def random_row(rng, nlimbs, kind):
    style = rng.randrange(4)
    if style == 0:
        limbs = [rng.getrandbits(64) for _ in range(nlimbs)]
    elif style == 1:
        limbs = [0] * nlimbs
        for _ in range(rng.randint(1, nlimbs)):
            limbs[rng.randrange(nlimbs)] = rng.getrandbits(64)
    elif style == 2:
        limbs = [rng.getrandbits(rng.choice([1, 4, 16, 32, 64])) for _ in range(nlimbs)]
    else:
        limbs = [0xFFFFFFFFFFFFFFFF if rng.getrandbits(1) else 0 for _ in range(nlimbs)]
    neg = kind != "unsigned" and bool(rng.getrandbits(1))
    return (limbs, neg)


def random_resolution(rng):
    style = rng.randrange(8)
    if style == 0:
        return 1.0
    if style == 1:
        return math.ldexp(1.0, rng.randrange(-64, 65))
    if style == 2:
        return float(rng.choice([0.25, 0.5, 2.5, 3.7, 0.1, 1e-3, 1e-6, 1e-9, 1234.5678, 1e6, 1e9]))
    if style == 3:
        return math.ldexp(1.0, rng.randrange(-1074, 1024))
    if style == 4:
        return float(rng.randint(1, 1 << 40))
    if style == 5:
        return math.ldexp(1.0, rng.randrange(-200, 201))
    return float(rng.random() * (10.0 ** rng.randrange(-9, 10)))


def build_rows(nlimbs, kind, rng):
    rows = structural_rows(nlimbs, kind)
    while len(rows) < ROWS:
        rows.append(random_row(rng, nlimbs, kind))
    return rows[:ROWS]


def row_values(exact, rng):
    """Pick a resolution whose scaled/divided results stay away from subnormals."""
    exact = Fraction(exact)
    for _ in range(40):
        resolution = random_resolution(rng)
        if resolution == 0.0:
            continue  # dividing by a resolution of zero is not defined
        scaled = exact_to_double(exact * Fraction(resolution))
        divided = exact_to_double(exact / Fraction(resolution))
        if all(r == 0.0 or abs(r) >= DBL_MIN for r in (scaled, divided)):
            return resolution, exact_to_double(exact), scaled, divided
    return 1.0, exact_to_double(exact), exact_to_double(exact), exact_to_double(exact)


def fmt_limb(value):
    return "0x%016XULL" % value


def fmt_double(value):
    if math.isinf(value):
        return ("std::numeric_limits<double>::infinity()" if value > 0
                else "-std::numeric_limits<double>::infinity()")
    if value == 0.0:
        return "0x0.0p+0"
    return "%s /* %.17g */" % (value.hex(), value)


def emit_type(cpp_type, test_id, limb_names, kind, rng):
    rows = build_rows(len(limb_names), kind, rng)
    lines = []
    lines.append("// ---------------------------------------------------------------------------")
    lines.append("// %s" % cpp_type)
    lines.append("// ---------------------------------------------------------------------------")
    lines.append("")
    lines.append("struct %sCase" % test_id)
    lines.append("{")
    for name in limb_names:
        lines.append("    uint64_t %s;" % name)
    if kind == "magnitude":
        lines.append("    bool neg;")
    lines.append("    double resolution;")
    lines.append("    double as_double;")
    lines.append("    double times_resolution;")
    lines.append("    double over_resolution;")
    lines.append("};")
    lines.append("")

    limbs_arg = ", ".join("c.%s" % name for name in limb_names)
    if kind == "magnitude" and cpp_type == "ext_sgn_int512_t":
        ctor = "ext_sgn_int512_t(uint512_t(%s), c.neg)" % limbs_arg
    elif kind == "magnitude":
        ctor = "%s(%s, c.neg)" % (cpp_type, limbs_arg)
    else:
        ctor = "%s(%s)" % (cpp_type, limbs_arg)

    lines.append("static %s make_value(const %sCase& c) { return %s; }" % (cpp_type, test_id, ctor))
    lines.append("")
    lines.append("// as_double / times_resolution / over_resolution are the correctly rounded")
    lines.append("// references computed by the generator (Python int and Fraction).")
    if kind == "twos":
        lines.append("// The limbs are the raw two's-complement bit pattern, so the top bit carries")
        lines.append("// the sign.")
    elif kind == "magnitude":
        lines.append("// The limbs are the unsigned magnitude; `neg` selects the sign.")
    lines.append("static const %sCase k%sCases[] = {" % (test_id, test_id))
    for limbs, neg in rows:
        stored, stored_neg, exact = normalize(limbs, neg, kind)
        resolution, plain, scaled, divided = row_values(exact, rng)
        fields = [fmt_limb(limb) for limb in stored]
        if kind == "magnitude":
            fields.append("true" if stored_neg else "false")
        fields.append(fmt_double(resolution))
        fields.append(fmt_double(plain))
        fields.append(fmt_double(scaled))
        fields.append(fmt_double(divided))
        lines.append("    {%s}," % ", ".join(fields))
    lines.append("};")
    lines.append("")
    lines.append("constexpr std::size_t k%sCaseCount = sizeof(k%sCases) / sizeof(k%sCases[0]);"
                 % (test_id, test_id, test_id))
    lines.append("")
    lines.append("// The limb-by-limb accumulation is within one ulp of the correctly rounded")
    lines.append("// reference; two ulps is a safety margin.")
    lines.append("constexpr double k%sUlpSlack = %.1f;" % (test_id, ULP_SLACK))
    lines.append("")

    for suffix, call, field in (
        ("ToDouble", "to_double(value)", "as_double"),
        ("ToDoubleScaledBy", "to_double_scaled_by(value, c.resolution)", "times_resolution"),
        ("ToDoubleDividedBy", "to_double_divided_by(value, c.resolution)", "over_resolution"),
    ):
        lines.append("TEST(BigIntFloat%s, %s)" % (test_id, suffix))
        lines.append("{")
        lines.append("    for (std::size_t i = 0; i < k%sCaseCount; ++i)" % test_id)
        lines.append("    {")
        lines.append("        const %sCase& c = k%sCases[i];" % (test_id, test_id))
        lines.append("        const %s value = make_value(c);" % cpp_type)
        lines.append("        expect_close_double(%s, c.%s, k%sUlpSlack," % (call, field, test_id))
        lines.append("                           std::string(\"%s case \") + std::to_string(i));" % cpp_type)
        lines.append("    }")
        lines.append("}")
        lines.append("")
    return lines


HEADER = """// AUTO-GENERATED by tests/Cuda/Test_Core/tools/gen_test_bigint_float.py -- do not edit.
//
// Random and extreme BigInt -> double conversion cases.  The expected values are
// computed with Python's arbitrary precision integers/Fractions and rounded once to
// the nearest double, so they are correctly rounded, implementation-independent
// references.  Every row is checked against to_double, to_double_scaled_by and
// to_double_divided_by; rows whose resolution is exactly 1 therefore also pin the
// two scaled paths to the plain conversion.
//
// Regenerate with: python tests/Cuda/Test_Core/tools/gen_test_bigint_float.py

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
// `max_ulps` units in the last place of `expected`.  The slack covers the rounding
// of the limb-by-limb accumulation, which runs in long double -- only 64 bits wide
// on MSVC and on the CUDA device.
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
"""


def main():
    rng = random.Random(SEED)
    out = [HEADER]
    for cpp_type, test_id, limb_names, kind in TYPES:
        out.extend(emit_type(cpp_type, test_id, limb_names, kind, rng))
        out.append("")

    text = "\n".join(out).rstrip() + "\n"
    target = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Test_BigIntFloat.cpp"))
    with open(target, "w", encoding="ascii", newline="\n") as handle:
        handle.write(text)
    print("wrote %s (%d lines)" % (target, text.count("\n")))


if __name__ == "__main__":
    main()
