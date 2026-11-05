#ifndef BIGINT_BIGINTFLOAT_H
#define BIGINT_BIGINTFLOAT_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include "BigInt.h"
#include "BigIntSigned.h"
#include "BigIntUtils.h"

// Bridge utilities to convert high-precision ints to floating point and do fp division.
//
// Everything here works in `double` only: `long double` is no wider than `double` on
// MSVC and on the CUDA device, so it buys no precision while making the header
// unusable in device code.
//
// Every BigInt type is converted limb by limb, most-significant limb first (Horner
// scheme).  Each step is an exact multiply by 2^64 followed by one rounded addition,
// so the result stays within a few ulp of the exact value even for a 512-bit input.
// No intermediate can overflow: double tops out near 2^1024 and the widest type,
// uint512_t, stays below 2^512.
namespace dyno
{
    // 2^64
    static constexpr double BIGINT_TWO64 = 18446744073709551616.0;

    inline DYN_FUNC double uint128_to_double(const uint128_t& x)
    {
        return static_cast<double>(x.hi) * BIGINT_TWO64 + static_cast<double>(x.lo);
    }

    inline DYN_FUNC float uint128_to_float(const uint128_t& x)
    {
        return static_cast<float>(uint128_to_double(x));
    }

    // Longhand division: compute at most one 64-bit fractional limb (enough for correct rounding in double).
    inline DYN_FUNC double uint128_div_uint64_longhand_to_double(const uint128_t& numerator, uint64_t denominator, unsigned extra_fraction_bits = 0)
    {
        (void)extra_fraction_bits; // extra precision beyond one limb does not improve double/float representation
        assert(denominator != 0);
        uint64_t q_hi = 0, q_lo = 0, rem = 0;
        div128by64_rem(numerator.hi, numerator.lo, denominator, q_hi, q_lo, rem);

        // Compose integer part carefully using ldexp to avoid intermediate overflow.
        double int_part = std::ldexp(static_cast<double>(q_hi), 64) + static_cast<double>(q_lo);
        if (rem == 0)
        {
            return int_part;
        }

        // Single 64-bit fractional block: (rem << 64) / denominator.
        uint64_t frac_q = 0, frac_rem = 0;
        div128by64_rem(rem, 0, denominator, frac_q, frac_rem);
        double frac_part = std::ldexp(static_cast<double>(frac_q), -64);

        // When the integer part is huge (>= 2^53), fractional contribution may be below double resolution.
        return int_part + frac_part;
    }

    inline DYN_FUNC float uint128_div_uint64_longhand_to_float(const uint128_t& numerator, uint64_t denominator, unsigned extra_fraction_bits = 0)
    {
        // For float, a single fractional limb is already more than enough for correct rounding.
        return static_cast<float>(uint128_div_uint64_longhand_to_double(numerator, denominator, extra_fraction_bits));
    }

    inline DYN_FUNC double uint128_div_to_double(const uint128_t& numerator, const uint128_t& denominator)
    {
        assert(denominator.hi != 0 || denominator.lo != 0);
        return uint128_to_double(numerator) / uint128_to_double(denominator);
    }

    inline DYN_FUNC float uint128_div_to_float(const uint128_t& numerator, const uint128_t& denominator)
    {
        assert(denominator.hi != 0 || denominator.lo != 0);
        // Divide in double: converting both operands to float first would overflow to
        // infinity for large values whose ratio is perfectly representable.
        return static_cast<float>(uint128_div_to_double(numerator, denominator));
    }

    // =====================================================================
    // BigInt -> double
    // =====================================================================

    // ---- unsigned magnitudes --------------------------------------------

    inline DYN_FUNC double to_double(const uint128_t& x)
    {
        return static_cast<double>(x.hi) * BIGINT_TWO64 + static_cast<double>(x.lo);
    }

    inline DYN_FUNC double to_double(const uint192_t& x)
    {
        double value = static_cast<double>(x.hi);
        value = value * BIGINT_TWO64 + static_cast<double>(x.mi);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo);
        return value;
    }

    inline DYN_FUNC double to_double(const uint256_t& x)
    {
        double value = static_cast<double>(x.hi1);
        value = value * BIGINT_TWO64 + static_cast<double>(x.hi0);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo1);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo0);
        return value;
    }

    inline DYN_FUNC double to_double(const uint512_t& x)
    {
        double value = static_cast<double>(x.lo7);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo6);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo5);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo4);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo3);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo2);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo1);
        value = value * BIGINT_TWO64 + static_cast<double>(x.lo0);
        return value;
    }

    // ---- signed, sign-magnitude encoding (neg_mask) ----------------------

    inline DYN_FUNC double to_double(const ext_sgn_int128_t& x)
    {
        const double magnitude = to_double(uint128_t(x.lo, x.hi));
        return (x.neg_mask != 0 && magnitude != 0.0) ? -magnitude : magnitude;
    }

    inline DYN_FUNC double to_double(const ext_sgn_int192_t& x)
    {
        const double magnitude = to_double(uint192_t(x.limb0, x.limb1, x.limb2));
        return (x.neg_mask != 0 && magnitude != 0.0) ? -magnitude : magnitude;
    }

    inline DYN_FUNC double to_double(const ext_sgn_int256_t& x)
    {
        const double magnitude = to_double(uint256_t(x.lo0, x.lo1, x.hi0, x.hi1));
        return (x.neg_mask != 0 && magnitude != 0.0) ? -magnitude : magnitude;
    }

    inline DYN_FUNC double to_double(const ext_sgn_int512_t& x)
    {
        const double magnitude = to_double(x.abs_val);
        return (x.neg_mask != 0 && magnitude != 0.0) ? -magnitude : magnitude;
    }

    // ---- signed, two's-complement encoding -------------------------------

    inline DYN_FUNC double to_double(const int128_t& x)
    {
        const double magnitude = to_double(x.abs_u());
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC double to_double(const int192_t& x)
    {
        const double magnitude = to_double(x.abs_u());
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC double to_double(const int256_t& x)
    {
        const double magnitude = to_double(x.abs_u());
        return x.is_negative() ? -magnitude : magnitude;
    }

    // ---- scaling by a resolution -----------------------------------------

    template <typename T>
    inline DYN_FUNC float to_float(const T& value)
    {
        return static_cast<float>(to_double(value));
    }

    // result = value * resolution, where `resolution` is the real-world size of one
    // integer unit (e.g. 1e-6).
    template <typename T>
    inline DYN_FUNC double to_double_scaled_by(const T& value, double resolution)
    {
        return to_double(value) * resolution;
    }

    // result = value / resolution, where `resolution` is the number of integer units
    // per real-world unit (e.g. 1e6).
    template <typename T>
    inline DYN_FUNC double to_double_divided_by(const T& value, double resolution)
    {
        assert(resolution != 0.0);
        return to_double(value) / resolution;
    }
}


#endif // BIGINT_BIGINTFLOAT_H