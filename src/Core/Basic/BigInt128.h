#ifndef BIGINT_BIGINT128_H
#define BIGINT_BIGINT128_H

#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include "BigIntConfig.h"
#include "BigIntUtils.h"

#ifndef DYN_FUNC
#define DYN_FUNC DYNO_HOST_DEVICE
#endif

namespace dyno
{
#if USE_PTX_DEPRECATED

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

    struct uint128_t
    {
        uint4 limbs; // x,y,z,w correspond to low to high 32-bit parts

        __host__ __device__ uint128_t()
            : limbs(make_uint4(0, 0, 0, 0))
        {
        }

        __host__ __device__ uint128_t(uint64_t lo, uint64_t hi)
            : limbs(make_uint4(static_cast<uint32_t>(lo), static_cast<uint32_t>(lo >> 32),
                                static_cast<uint32_t>(hi), static_cast<uint32_t>(hi >> 32)))
        {
        }

        __device__ uint128_t operator+(const uint128_t& other) const
        {
            uint128_t result;
            add_uint128_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ uint128_t operator-(const uint128_t& other) const
        {
            uint128_t result;
            sub_uint128_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ uint128_t operator*(const uint128_t& other) const
        {
            uint128_t result;
            mul_uint128_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ bool operator>=(const uint128_t& other) const
        {
            if (limbs.w != other.limbs.w) return limbs.w > other.limbs.w;
            if (limbs.z != other.limbs.z) return limbs.z > other.limbs.z;
            if (limbs.y != other.limbs.y) return limbs.y > other.limbs.y;
            return limbs.x >= other.limbs.x;
        }

        __device__ bool operator<(const uint128_t& other) const
        {
            return !(*this >= other);
        }

        __device__ bool operator==(const uint128_t& other) const = default;
    };

#else // !USE_PTX

    struct uint128_t
    {
        uint64_t lo;
        uint64_t hi;

        DYN_FUNC uint128_t() = default;
        DYN_FUNC uint128_t(uint64_t lo, uint64_t hi) : lo(lo), hi(hi) {}

        DYN_FUNC uint128_t operator+(const uint128_t& other) const
        {
            uint128_t result;
            result.lo = lo + other.lo;
            result.hi = hi + other.hi + (result.lo < lo ? 1 : 0);
            return result;
        }

        static DYN_FUNC uint128_t from_uint64_mul(uint64_t a, uint64_t b)
        {
            uint128_t res;
            mul64x64_to_128(a, b, res.lo, res.hi);
            return res;
        }

        static DYN_FUNC uint128_t from_uint32_mul(uint32_t a, uint32_t b, uint32_t c)
        {
            uint128_t res;
            mul64x32_to_128(a * b, c, res.lo, res.hi);
            return res;
        }

        static DYN_FUNC uint128_t from_uint32_mul(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
        {
            uint128_t res;
            mul64x64_to_128(uint64_t(a) * b, uint64_t(c) * d, res.lo, res.hi);
            return res;
        }

        DYN_FUNC uint128_t operator-(const uint128_t& other) const
        {
            uint128_t result;
            result.lo = lo - other.lo;
            result.hi = hi - other.hi - (lo < other.lo ? 1 : 0);
            return result;
        }

        DYN_FUNC uint128_t operator*(const uint128_t& other) const
        {
            uint128_t result;
            uint64_t lo_lo, lo_hi, hi_lo, hi_hi;

            dyno::mul64x64_to_128(lo, other.lo, lo_lo, lo_hi);
            dyno::mul64x64_to_128(hi, other.lo, hi_lo, hi_hi);

            uint64_t tmp = lo_hi + hi_lo;
            uint64_t carry = (tmp < lo_hi) ? 1 : 0;

            dyno::mul64x64_to_128(lo, other.hi, hi_lo, hi_hi);

            uint64_t tmp2 = tmp + hi_lo;
            carry += (tmp2 < tmp) ? 1 : 0;

            result.lo = lo_lo;
            result.hi = tmp2;

            return result;
        }

        DYN_FUNC uint128_t operator/(uint64_t other) const
        {
            uint128_t res;
            div128by64(hi, lo, other, res.hi, res.lo);
            return res;
        }

        DYN_FUNC uint64_t operator%(uint64_t other) const
        {
            uint64_t q, r;
            div128by64_rem(hi, lo, other, q, r);
            return r;
        }

        DYN_FUNC uint128_t operator<<(int shift) const
        {
            if (shift == 0)
            {
                return *this;
            }
            if (shift < 64)
            {
                uint64_t new_hi = (hi << shift) | (lo >> (64 - shift));
                uint64_t new_lo = lo << shift;
                return {new_lo, new_hi};
            }
            if (shift < 128)
            {
                uint64_t new_hi = lo << (shift - 64);
                uint64_t new_lo = 0;
                return {new_lo, new_hi};
            }
            return {0, 0};
        }

        // Try to add other to this, if overflow occurs, return false
        DYN_FUNC bool try_add(const uint128_t& other, uint128_t& res) const
        {
            auto new_lo = lo + other.lo;
            auto carry = (new_lo < lo) ? 1 : 0;
            auto new_hi = hi + other.hi + carry;
            if (new_hi < hi || (carry == 1 && new_hi == hi))
                return false;
            res.lo = new_lo;
            res.hi = new_hi;
            return true;
        }

        DYN_FUNC bool try_sub(const uint128_t& other, uint128_t& res) const
        {
            if (*this < other)
                return false;
            res.lo = lo - other.lo;
            res.hi = hi - other.hi - (lo < other.lo ? 1 : 0);
            return true;
        }

        DYN_FUNC bool try_mul(const uint128_t& other, uint128_t& res) const
        {
            if (hi != 0 && other.hi != 0)
                return false;
            uint64_t lo_lo, lo_hi, hi_lo, hi_hi;

            dyno::mul64x64_to_128(lo, other.lo, lo_lo, lo_hi);
            dyno::mul64x64_to_128(hi, other.lo, hi_lo, hi_hi);

            uint64_t tmp = lo_hi + hi_lo;
            uint64_t carry = (tmp < lo_hi) ? 1 : 0;

            dyno::mul64x64_to_128(lo, other.hi, hi_lo, hi_hi);

            if (addition_will_overflow(tmp, hi_lo))
                return false;
            uint64_t tmp2 = tmp + hi_lo;

            res.lo = lo_lo;
            res.hi = tmp2;

            return true;
        }

        DYN_FUNC uint128_t operator/(uint128_t other) const
        {
            if (*this < other)
            {
                return {0, 0};
            }

            if (other.hi == 0)
            {
                uint128_t res;
                div128by64(hi, lo, other.lo, res.hi, res.lo);
                return res;
            }

            int s = count_leading_zeros(other.hi); // 0 <= s < 64
            uint128_t vn = other << s;

            uint128_t un_lo = (*this) << s;
            uint64_t un2 = (s == 0) ? 0 : (hi >> (64 - s));
            uint64_t un1 = un_lo.hi;
            uint64_t un0 = un_lo.lo;

            uint64_t vn1 = vn.hi;
            uint64_t vn0 = vn.lo;

            // estimate qhat = floor((un2*b + un1)/vn1)
            uint64_t rem = 0;
            uint64_t qhat;
            div128by64_rem(un2, un1, vn1, qhat, rem);
            uint64_t rhat = rem;

            while (from_uint64_mul(qhat, vn0) > uint128_t{un0, rhat})
            {
                qhat--;
                uint64_t old = rhat;
                rhat += vn1;
                if (rhat < old) break;
            }

            uint64_t qv0_lo, qv0_hi;
            mul64x64_to_128(qhat, vn0, qv0_lo, qv0_hi);
            uint64_t qv1_lo, qv1_hi;
            mul64x64_to_128(qhat, vn1, qv1_lo, qv1_hi);

            uint64_t p0 = qv0_lo;
            uint64_t p1, p2;
            add128(qv1_hi, qv0_hi, 0, qv1_lo, p2, p1);

            int cmp = cmp_3_limbs(un2, un1, un0, p2, p1, p0);

            if (cmp < 0)
            {
                qhat--;
            }

            uint128_t q = {qhat, 0};
            return q;
        }


        DYN_FUNC uint128_t operator%(uint128_t other) const
        {
            // Simpler: reuse division result; remainder = a - q * b
            uint128_t q = (*this) / other;
            uint128_t prod = q * other;
            return (*this) - prod;
        }

        
        DYN_FUNC uint128_t operator>>(int shift) const
        {
            if (shift == 0)
            {
                return *this;
            }
            if (shift < 64)
            {
                uint64_t new_lo = (lo >> shift) | (hi << (64 - shift));
                uint64_t new_hi = hi >> shift;
                return {new_lo, new_hi};
            }
            if (shift < 128)
            {
                uint64_t new_lo = hi >> (shift - 64);
                uint64_t new_hi = 0;
                return {new_lo, new_hi};
            }
            return {0, 0};
        }

        DYN_FUNC bool operator>=(const uint128_t& other) const
        {
            return (hi > other.hi) || (hi == other.hi && lo >= other.lo);
        }

        DYN_FUNC bool operator<(const uint128_t& other) const
        {
            return (hi < other.hi) || (hi == other.hi && lo < other.lo);
        }

        DYN_FUNC bool operator>(const uint128_t& other) const
        {
            return this->hi > other.hi || (this->hi == other.hi && this->lo > other.lo);
        }

        DYN_FUNC bool operator==(const uint128_t& uint128) const
        {
            return (this->hi == uint128.hi) && (this->lo == uint128.lo);
        }
    };

    DYN_FUNC inline uint64_t estimate_qhat(uint64_t un2, uint64_t un1, uint64_t un0, uint64_t vn1, uint64_t vn0)
    {
        uint64_t qhat, rem;
        if (un2 == 0)
        {
            qhat = un1 / vn1;
            rem = un1 % vn1;
        }
        else
        {
            div128by64_rem(un2, un1, vn1, qhat, rem);
        }
        uint64_t rhat = rem;
        while (uint128_t::from_uint64_mul(qhat, vn0) > uint128_t{un0, rhat})
        {
            qhat--;
            uint64_t old = rhat;
            rhat += vn1;
            if (rhat < old) break;
        }
        return qhat;
    }

    struct ext_sgn_int128_t
    {
        uint64_t lo;
        uint64_t hi;
        uint64_t neg_mask;

        DYN_FUNC ext_sgn_int128_t() : lo(0), hi(0), neg_mask(0) {}

        DYN_FUNC ext_sgn_int128_t(uint64_t low, uint64_t high, bool negative)
            : lo(low), hi(high), neg_mask(negative ? ~0ULL : 0) {}

        DYN_FUNC ext_sgn_int128_t(int64_t a)
            : hi(0)
        {
            neg_mask = (a < 0) ? ~0ULL : 0;
            lo = static_cast<uint64_t>(a < 0 ? uint64_t(-(a + (a == INT64_MIN))) + (a == INT64_MIN) : a);
        }

        DYN_FUNC static ext_sgn_int128_t from_int32_mul(int32_t a, int32_t b, int32_t c)

        {
            bool negative = ((a < 0) ^ (b < 0)) ^ (c < 0);
            uint32_t abs_a = std::abs(a);
            uint32_t abs_b = std::abs(b);
            uint32_t abs_c = std::abs(c);
            uint64_t lo, hi;
            dyno::mul64x32_to_128(uint64_t(abs_a) * abs_b, abs_c, lo, hi);
            return ext_sgn_int128_t(lo, hi, negative);
        }

        DYN_FUNC static ext_sgn_int128_t from_int32_mul(int32_t a, int32_t b, int32_t c, int32_t d)
        {
            bool negative = ((a < 0) ^ (b < 0)) ^ ((c < 0) ^ (d < 0));
            uint32_t abs_a = std::abs(a);
            uint32_t abs_b = std::abs(b);
            uint32_t abs_c = std::abs(c);
            uint32_t abs_d = std::abs(d);
            uint64_t lo, hi;
            dyno::mul64x64_to_128(uint64_t(abs_a) * abs_b, uint64_t(abs_c) * abs_d, lo, hi);
            return ext_sgn_int128_t(lo, hi, negative);
        }

        DYN_FUNC static ext_sgn_int128_t from_int64_mul(int64_t a, int64_t b)
        {
            bool negative = (a < 0) ^ (b < 0);

            uint64_t ua = static_cast<uint64_t>(a < 0 ? uint64_t(-(a + (a == INT64_MIN))) + (a == INT64_MIN) : a);
            uint64_t ub = static_cast<uint64_t>(b < 0 ? uint64_t(-(b + (b == INT64_MIN))) + (b == INT64_MIN) : b);

            uint64_t lo, hi;
            dyno::mul64x64_to_128(ua, ub, lo, hi);

            return ext_sgn_int128_t(lo, hi, negative);
        }

        DYN_FUNC bool try_add(const ext_sgn_int128_t& b, ext_sgn_int128_t& res) const
        {
            uint128_t abs_a = {lo, hi};
            uint128_t abs_b = {b.lo, b.hi};
            if (neg_mask == b.neg_mask)
            {
                uint128_t sum;
                if (!abs_a.try_add(abs_b, sum))
                    return false;
                res.lo = sum.lo;
                res.hi = sum.hi;
                res.neg_mask = neg_mask;
                return true;
            }
            uint128_t diff;
            if (!abs_a.try_sub(abs_b, diff))
                return false;
            bool a_ge_b = (hi > b.hi) || (hi == b.hi && lo >= b.lo);
            res.lo = diff.lo;
            res.hi = diff.hi;
            res.neg_mask = a_ge_b ? neg_mask : b.neg_mask;
            return true;
        }

        DYN_FUNC bool try_sub(const ext_sgn_int128_t& b, ext_sgn_int128_t& res) const
        {
            ext_sgn_int128_t neg_b = b;
            neg_b.neg_mask = ~neg_b.neg_mask;
            return this->try_add(neg_b, res);
        }

        DYN_FUNC bool try_mul(const ext_sgn_int128_t& b, ext_sgn_int128_t& res) const
        {
            uint128_t abs_a = {lo, hi};
            uint128_t abs_b = {b.lo, b.hi};
            uint128_t prod;
            if (!abs_a.try_mul(abs_b, prod))
                return false;
            res.lo = prod.lo;
            res.hi = prod.hi;
            res.neg_mask = (neg_mask ^ b.neg_mask);
            return true;
        }

        DYN_FUNC ext_sgn_int128_t operator+(const ext_sgn_int128_t& b) const
        {
            uint64_t same_sign = ~(neg_mask ^ b.neg_mask);
            uint64_t diff_sign = neg_mask ^ b.neg_mask;

            uint64_t sum_lo, sum_hi;
            dyno::add128(hi, lo, b.hi, b.lo, sum_hi, sum_lo);

            uint64_t a_ge_b = ((hi > b.hi) || ((hi == b.hi) && (lo >= b.lo))) ? ~0ULL : 0;
            uint64_t diff_lo, diff_hi;
            uint64_t hi1 = hi, lo1 = lo;
            uint64_t hi2 = b.hi, lo2 = b.lo;
            uint64_t mask_hi = a_ge_b;
            uint64_t mask_lo = ~a_ge_b;
            uint64_t sub_hi = (hi1 & mask_hi) | (hi2 & mask_lo);
            uint64_t sub_lo = (lo1 & mask_hi) | (lo2 & mask_lo);
            uint64_t other_hi = (hi1 & mask_lo) | (hi2 & mask_hi);
            uint64_t other_lo = (lo1 & mask_lo) | (lo2 & mask_hi);
            dyno::sub128(sub_hi, sub_lo, other_hi, other_lo, diff_hi, diff_lo);

            uint64_t res_lo = (sum_lo & same_sign) | (diff_lo & diff_sign);
            uint64_t res_hi = (sum_hi & same_sign) | (diff_hi & diff_sign);

            uint64_t res_neg = (same_sign & neg_mask) | (diff_sign & ((a_ge_b & neg_mask) | (~a_ge_b & b.neg_mask)));
            if ((res_lo | res_hi) == 0) res_neg = 0;

            return ext_sgn_int128_t(res_lo, res_hi, res_neg != 0);
        }


        DYN_FUNC ext_sgn_int128_t operator-(const ext_sgn_int128_t& b) const
        {
            ext_sgn_int128_t neg_b = b;
            neg_b.neg_mask = ~neg_b.neg_mask;
            return *this + neg_b;
        }

        DYN_FUNC ext_sgn_int128_t operator-() const
        {
            if (lo == 0 && hi == 0)
            {
                return ext_sgn_int128_t(0, 0, false);
            }
            return ext_sgn_int128_t(lo, hi, neg_mask == 0);
        }

        DYN_FUNC bool operator<(const ext_sgn_int128_t& other) const
        {
            bool zero_this = (lo == 0 && hi == 0);
            bool zero_other = (other.lo == 0 && other.hi == 0);

            uint64_t eff_neg_mask = zero_this ? 0 : neg_mask;
            uint64_t eff_other_neg_mask = zero_other ? 0 : other.neg_mask;

            uint64_t this_is_neg_other_is_pos = eff_neg_mask & ~eff_other_neg_mask;

            uint64_t abs_greater = ((hi > other.hi) || (hi == other.hi && lo > other.lo)) ? ~0ULL : 0;
            uint64_t abs_equal = ((hi == other.hi) && (lo == other.lo)) ? ~0ULL : 0;

            uint64_t both_pos_mask = ~eff_neg_mask & ~eff_other_neg_mask;
            uint64_t lt_if_both_pos = both_pos_mask & ~abs_greater & ~abs_equal;

            uint64_t both_neg_mask = eff_neg_mask & eff_other_neg_mask;
            uint64_t lt_if_both_neg = both_neg_mask & abs_greater;

            return (this_is_neg_other_is_pos | lt_if_both_pos | lt_if_both_neg) != 0;
        }


        DYN_FUNC int sgn() const
        {
            if (hi == 0 && lo == 0)
                return 0;
            return neg_mask ? -1 : 1;
        }

        DYN_FUNC ext_sgn_int128_t operator*(const ext_sgn_int128_t& other) const
        {
            uint128_t a = {lo, hi};
            uint128_t b = {other.lo, other.hi};
            uint128_t res = a * b;
            bool res_neg = (neg_mask ^ other.neg_mask) != 0;
            return ext_sgn_int128_t(res.lo, res.hi, res_neg);
        }

        DYN_FUNC bool operator<=(const ext_sgn_int128_t& other) const
        {
            return !(*this > other);
        }

        DYN_FUNC bool operator>(const ext_sgn_int128_t& other) const
        {
            return other < *this;
        }

        DYN_FUNC bool operator>=(const ext_sgn_int128_t& other) const
        {
            return !(*this < other);
        }

        DYN_FUNC bool operator==(const ext_sgn_int128_t& other) const
        {
            bool zero_this = (lo == 0 && hi == 0);
            bool zero_other = (other.lo == 0 && other.hi == 0);
            if (zero_this && zero_other) return true;
            return (lo == other.lo) && (hi == other.hi) && (neg_mask == other.neg_mask);
        }


        DYN_FUNC bool operator!=(const ext_sgn_int128_t& other) const
        {
            return !(*this == other);
        }

        DYN_FUNC ext_sgn_int128_t operator/(const ext_sgn_int128_t& other) const
        {
            // Handle division by zero
            if (other.lo == 0 && other.hi == 0)
            {
                return ext_sgn_int128_t(~0ULL, ~0ULL, true);
            }

            uint128_t abs_this = {lo, hi};
            uint128_t abs_other = {other.lo, other.hi};

            uint128_t abs_result = abs_this / abs_other;

            bool result_negative = (neg_mask ^ other.neg_mask) != 0;

            return ext_sgn_int128_t(abs_result.lo, abs_result.hi, result_negative);
        }

        DYN_FUNC ext_sgn_int128_t operator%(const ext_sgn_int128_t& other) const
        {
            // Handle division by zero
            if (other.lo == 0 && other.hi == 0)
            {
                return ext_sgn_int128_t(~0ULL, ~0ULL, true);
            }

            uint128_t abs_this = {lo, hi};
            uint128_t abs_other = {other.lo, other.hi};

            uint128_t abs_rem = abs_this % abs_other;
            bool rem_is_zero = (abs_rem.lo == 0 && abs_rem.hi == 0);
            bool result_negative = (neg_mask != 0) && !rem_is_zero;

            return ext_sgn_int128_t(abs_rem.lo, abs_rem.hi, result_negative);
        }
        DYN_FUNC uint128_t abs() const
        {
            return uint128_t{lo,hi};
        }

    };


#endif // DYNO_ENABLE_PTX
} // namespace dyno

#endif // BIGINT_BIGINT128_H
