#ifndef BIGINT_BIGINTFLOAT_H
#define BIGINT_BIGINTFLOAT_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include "BigInt.h"
#include "BigIntSigned.h"
#include "BigIntUtils.h"

// Bridge utilities to convert high-precision unsigned/signed ints to floating point and do fp division.
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


    // =====================================================================
    // Generic BigInt -> floating point conversion
    //
    // Every conversion walks the limbs explicitly, most-significant limb first
    // (Horner scheme), so the accumulated rounding error stays within a couple
    // of ulp of the intermediate type even when the value far exceeds its
    // mantissa width (e.g. uint512_t).
    //
    // The resolution is folded into every limb before the accumulation instead
    // of being applied to the finished value: that keeps the running value
    // close to the final magnitude, so a 512-bit integer can still be scaled
    // back into double range even though long double is only 64 bits wide on
    // MSVC and CUDA (converting the value first would overflow to infinity).
    // =====================================================================

    static constexpr long double BIGINT_TWO64 = 18446744073709551616.0L; // 2^64

    // ---- uint128_t ------------------------------------------------------

    inline DYN_FUNC long double to_long_double(const uint128_t& x)
    {
        return static_cast<long double>(x.hi) * BIGINT_TWO64 + static_cast<long double>(x.lo);
    }

    // value * resolution
    inline DYN_FUNC long double to_long_double_scaled_by(const uint128_t& x, long double resolution)
    {
        long double value = static_cast<long double>(x.hi) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo) * resolution;
        return value;
    }

    // value / resolution
    inline DYN_FUNC long double to_long_double_divided_by(const uint128_t& x, long double resolution)
    {
        assert(resolution != 0.0L);
        long double value = static_cast<long double>(x.hi) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo) / resolution;
        return value;
    }

    // ---- uint192_t ------------------------------------------------------

    inline DYN_FUNC long double to_long_double(const uint192_t& x)
    {
        long double value = static_cast<long double>(x.hi);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.mi);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo);
        return value;
    }

    // value * resolution
    inline DYN_FUNC long double to_long_double_scaled_by(const uint192_t& x, long double resolution)
    {
        long double value = static_cast<long double>(x.hi) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.mi) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo) * resolution;
        return value;
    }

    // value / resolution
    inline DYN_FUNC long double to_long_double_divided_by(const uint192_t& x, long double resolution)
    {
        assert(resolution != 0.0L);
        long double value = static_cast<long double>(x.hi) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.mi) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo) / resolution;
        return value;
    }

    // ---- uint256_t ------------------------------------------------------

    inline DYN_FUNC long double to_long_double(const uint256_t& x)
    {
        long double value = static_cast<long double>(x.hi1);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.hi0);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo1);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo0);
        return value;
    }

    // value * resolution
    inline DYN_FUNC long double to_long_double_scaled_by(const uint256_t& x, long double resolution)
    {
        long double value = static_cast<long double>(x.hi1) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.hi0) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo1) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo0) * resolution;
        return value;
    }

    // value / resolution
    inline DYN_FUNC long double to_long_double_divided_by(const uint256_t& x, long double resolution)
    {
        assert(resolution != 0.0L);
        long double value = static_cast<long double>(x.hi1) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.hi0) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo1) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo0) / resolution;
        return value;
    }

    // ---- uint512_t ------------------------------------------------------

    inline DYN_FUNC long double to_long_double(const uint512_t& x)
    {
        long double value = static_cast<long double>(x.lo7);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo6);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo5);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo4);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo3);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo2);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo1);
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo0);
        return value;
    }

    // value * resolution
    inline DYN_FUNC long double to_long_double_scaled_by(const uint512_t& x, long double resolution)
    {
        long double value = static_cast<long double>(x.lo7) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo6) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo5) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo4) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo3) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo2) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo1) * resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo0) * resolution;
        return value;
    }

    // value / resolution
    inline DYN_FUNC long double to_long_double_divided_by(const uint512_t& x, long double resolution)
    {
        assert(resolution != 0.0L);
        long double value = static_cast<long double>(x.lo7) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo6) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo5) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo4) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo3) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo2) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo1) / resolution;
        value = value * BIGINT_TWO64 + static_cast<long double>(x.lo0) / resolution;
        return value;
    }

    // ---- signed, sign-magnitude encoding (neg_mask) ----------------------

    inline DYN_FUNC long double to_long_double(const ext_sgn_int128_t& x)
    {
        const long double magnitude = to_long_double(uint128_t{x.lo, x.hi});
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_scaled_by(const ext_sgn_int128_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_scaled_by(uint128_t{x.lo, x.hi}, resolution);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_divided_by(const ext_sgn_int128_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_divided_by(uint128_t{x.lo, x.hi}, resolution);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double(const ext_sgn_int192_t& x)
    {
        const long double magnitude = to_long_double(uint192_t(x.limb0, x.limb1, x.limb2));
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_scaled_by(const ext_sgn_int192_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_scaled_by(uint192_t(x.limb0, x.limb1, x.limb2), resolution);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_divided_by(const ext_sgn_int192_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_divided_by(uint192_t(x.limb0, x.limb1, x.limb2), resolution);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double(const ext_sgn_int256_t& x)
    {
        const long double magnitude = to_long_double(uint256_t(x.lo0, x.lo1, x.hi0, x.hi1));
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_scaled_by(const ext_sgn_int256_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_scaled_by(uint256_t(x.lo0, x.lo1, x.hi0, x.hi1), resolution);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_divided_by(const ext_sgn_int256_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_divided_by(uint256_t(x.lo0, x.lo1, x.hi0, x.hi1), resolution);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double(const ext_sgn_int512_t& x)
    {
        const long double magnitude = to_long_double(x.abs_val);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_scaled_by(const ext_sgn_int512_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_scaled_by(x.abs_val, resolution);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_divided_by(const ext_sgn_int512_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_divided_by(x.abs_val, resolution);
        return (x.neg_mask != 0 && magnitude != 0.0L) ? -magnitude : magnitude;
    }

    // ---- signed, two's-complement encoding -------------------------------

    inline DYN_FUNC long double to_long_double(const int128_t& x)
    {
        const long double magnitude = to_long_double(x.abs_u());
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_scaled_by(const int128_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_scaled_by(x.abs_u(), resolution);
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_divided_by(const int128_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_divided_by(x.abs_u(), resolution);
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double(const int192_t& x)
    {
        const long double magnitude = to_long_double(x.abs_u());
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_scaled_by(const int192_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_scaled_by(x.abs_u(), resolution);
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_divided_by(const int192_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_divided_by(x.abs_u(), resolution);
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double(const int256_t& x)
    {
        const long double magnitude = to_long_double(x.abs_u());
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_scaled_by(const int256_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_scaled_by(x.abs_u(), resolution);
        return x.is_negative() ? -magnitude : magnitude;
    }

    inline DYN_FUNC long double to_long_double_divided_by(const int256_t& x, long double resolution)
    {
        const long double magnitude = to_long_double_divided_by(x.abs_u(), resolution);
        return x.is_negative() ? -magnitude : magnitude;
    }

    // ---- public helpers --------------------------------------------------

    template <typename T>
    inline DYN_FUNC double to_double(const T& value)
    {
        return static_cast<double>(to_long_double(value));
    }

    template <typename T>
    inline DYN_FUNC float to_float(const T& value)
    {
        return static_cast<float>(to_long_double(value));
    }

    // Scale by a resolution: result = value * resolution, where `resolution` is
    // the real-world size of one integer unit (e.g. 1e-6).
    template <typename T>
    inline DYN_FUNC double to_double_scaled_by(const T& value, double resolution)
    {
        return static_cast<double>(to_long_double_scaled_by(value, static_cast<long double>(resolution)));
    }

    // Inverse scaling: result = value / resolution, where `resolution` is the
    // number of integer units per real-world unit (e.g. 1e6).
    template <typename T>
    inline DYN_FUNC double to_double_divided_by(const T& value, double resolution)
    {
        assert(resolution != 0.0);
        return static_cast<double>(to_long_double_divided_by(value, static_cast<long double>(resolution)));
    }
}


#endif // BIGINT_BIGINTFLOAT_H
