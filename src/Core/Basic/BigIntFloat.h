#ifndef BIGINT_BIGINTFLOAT_H
#define BIGINT_BIGINTFLOAT_H

#include <cassert>
#include <cmath>
#include "BigInt.h"
#include "BigIntUtils.h"

// Bridge utilities to convert high-precision unsigned ints to floating point and do fp division.
namespace dyno
{
    // Convert uint128 to long double (keeps ~64+ bits of precision; exponent is ample for double/float division)
    inline DYN_FUNC long double uint128_to_long_double(const uint128_t& x)
    {
        static constexpr long double TWO64 = 18446744073709551616.0L; // 2^64
        return static_cast<long double>(x.hi) * TWO64 + static_cast<long double>(x.lo);
    }

    inline DYN_FUNC double uint128_to_double(const uint128_t& x)
    {
        return static_cast<double>(uint128_to_long_double(x));
    }

    inline DYN_FUNC float uint128_to_float(const uint128_t& x)
    {
        return static_cast<float>(uint128_to_long_double(x));
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
        long double num = uint128_to_long_double(numerator);
        long double den = uint128_to_long_double(denominator);
        return static_cast<double>(num / den);
    }

    inline DYN_FUNC float uint128_div_to_float(const uint128_t& numerator, const uint128_t& denominator)
    {
        assert(denominator.hi != 0 || denominator.lo != 0);
        long double num = uint128_to_long_double(numerator);
        long double den = uint128_to_long_double(denominator);
        return static_cast<float>(num / den);
    }
}


#endif // BIGINT_BIGINTFLOAT_H