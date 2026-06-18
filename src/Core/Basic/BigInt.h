#ifndef BIGINT_BIGINT_H
#define BIGINT_BIGINT_H

#include <cstdint>
#include <algorithm>
#include "BigIntConfig.h"
#include "BigIntUtils.h"

#ifndef DYN_FUNC
#define DYN_FUNC DYNO_HOST_DEVICE
#endif

#include "BigInt128.h"

namespace dyno
{
#if !DYNO_ENABLE_PTX

    struct uint192_t
    {
        uint64_t lo;
        uint64_t mi;
        uint64_t hi;

        DYN_FUNC uint192_t() : lo(0), mi(0), hi(0)
        {
        }

        DYN_FUNC uint192_t(uint64_t l, uint64_t m, uint64_t h) : lo(l), mi(m), hi(h)
        {
        }

        DYN_FUNC uint192_t operator+(const uint192_t& other) const
        {
            uint192_t result;
            result.lo = lo + other.lo;
            uint64_t carry = (result.lo < lo) ? 1 : 0;

            result.mi = mi + other.mi + carry;
            carry = (result.mi < mi || (carry == 1 && result.mi == mi)) ? 1 : 0;

            result.hi = hi + other.hi + carry;
            return result;
        }

        DYN_FUNC bool try_add(const uint192_t& other, uint192_t& res) const
        {
            auto new_lo = lo + other.lo;
            auto carry = (new_lo < lo) ? 1 : 0;
            auto new_mi = mi + other.mi + carry;
            carry = (new_mi < mi || (carry == 1 && new_mi == mi)) ? 1 : 0;
            auto new_hi = hi + other.hi + carry;
            if (addition_will_overflow(hi, other.hi, carry))
                return false;
            res.lo = new_lo;
            res.mi = new_mi;
            res.hi = new_hi;
            return true;
        }

        DYN_FUNC bool try_sub(const uint192_t& other, uint192_t& result) const
        {
            if (*this < other)
                return false;
            result = *this - other;
            return true;
        }

        DYN_FUNC bool try_mul(const uint192_t& other, uint192_t& result) const
        {
            uint64_t a0 = lo, a1 = mi, a2 = hi;
            uint64_t b0 = other.lo, b1 = other.mi, b2 = other.hi;

            uint64_t p00_lo, p00_hi;
            dyno::mul64x64_to_128(a0, b0, p00_lo, p00_hi);

            uint64_t p01_lo, p01_hi;
            dyno::mul64x64_to_128(a0, b1, p01_lo, p01_hi);

            uint64_t p02_lo, p02_hi;
            dyno::mul64x64_to_128(a0, b2, p02_lo, p02_hi);

            uint64_t p10_lo, p10_hi;
            dyno::mul64x64_to_128(a1, b0, p10_lo, p10_hi);

            uint64_t p11_lo, p11_hi;
            dyno::mul64x64_to_128(a1, b1, p11_lo, p11_hi);

            uint64_t p12_lo, p12_hi;
            dyno::mul64x64_to_128(a1, b2, p12_lo, p12_hi);

            uint64_t p20_lo, p20_hi;
            dyno::mul64x64_to_128(a2, b0, p20_lo, p20_hi);

            uint64_t p21_lo, p21_hi;
            dyno::mul64x64_to_128(a2, b1, p21_lo, p21_hi);

            uint64_t p22_lo, p22_hi;
            dyno::mul64x64_to_128(a2, b2, p22_lo, p22_hi);

            result.lo = p00_lo;

            uint64_t sum1 = p00_hi;
            sum1 += p01_lo;
            uint64_t carry1 = (sum1 < p00_hi) ? 1 : 0;
            sum1 += p10_lo;
            carry1 += (sum1 < p10_lo) ? 1 : 0;

            result.mi = sum1;

            uint64_t sum2 = carry1;
            sum2 += p01_hi;
            uint64_t carry2 = (sum2 < carry1) ? 1 : 0;
            sum2 += p02_lo;
            carry2 += (sum2 < p02_lo) ? 1 : 0;
            sum2 += p10_hi;
            carry2 += (sum2 < p10_hi) ? 1 : 0;
            sum2 += p11_lo;
            carry2 += (sum2 < p11_lo) ? 1 : 0;
            sum2 += p20_lo;
            carry2 += (sum2 < p20_lo) ? 1 : 0;

            if (carry2)
                return false;
            result.hi = sum2;
            return true;
        }

        DYN_FUNC uint192_t operator-(const uint192_t& other) const
        {
            uint192_t result;
            result.lo = lo - other.lo;
            uint64_t borrow = (lo < other.lo) ? 1 : 0;

            result.mi = mi - other.mi - borrow;
            borrow = (mi < other.mi || (borrow == 1 && mi == other.mi)) ? 1 : 0;

            result.hi = hi - other.hi - borrow;
            return result;
        }

        DYN_FUNC uint192_t operator*(const uint192_t& other) const
        {
            uint192_t result;
            uint64_t a0 = lo, a1 = mi, a2 = hi;
            uint64_t b0 = other.lo, b1 = other.mi, b2 = other.hi;

            uint64_t p00_lo, p00_hi;
            dyno::mul64x64_to_128(a0, b0, p00_lo, p00_hi);

            uint64_t p01_lo, p01_hi;
            dyno::mul64x64_to_128(a0, b1, p01_lo, p01_hi);

            uint64_t p02_lo, p02_hi;
            dyno::mul64x64_to_128(a0, b2, p02_lo, p02_hi);

            uint64_t p10_lo, p10_hi;
            dyno::mul64x64_to_128(a1, b0, p10_lo, p10_hi);

            uint64_t p11_lo, p11_hi;
            dyno::mul64x64_to_128(a1, b1, p11_lo, p11_hi);

            uint64_t p12_lo, p12_hi;
            dyno::mul64x64_to_128(a1, b2, p12_lo, p12_hi);

            uint64_t p20_lo, p20_hi;
            dyno::mul64x64_to_128(a2, b0, p20_lo, p20_hi);

            uint64_t p21_lo, p21_hi;
            dyno::mul64x64_to_128(a2, b1, p21_lo, p21_hi);

            uint64_t p22_lo, p22_hi;
            dyno::mul64x64_to_128(a2, b2, p22_lo, p22_hi);

            result.lo = p00_lo;

            uint64_t sum1 = p00_hi;
            sum1 += p01_lo;
            uint64_t carry1 = (sum1 < p00_hi) ? 1 : 0;
            sum1 += p10_lo;
            carry1 += (sum1 < p10_lo) ? 1 : 0;

            result.mi = sum1;

            uint64_t sum2 = carry1;
            sum2 += p01_hi;
            uint64_t carry2 = (sum2 < carry1) ? 1 : 0;
            sum2 += p02_lo;
            carry2 += (sum2 < p02_lo) ? 1 : 0;
            sum2 += p10_hi;
            carry2 += (sum2 < p10_hi) ? 1 : 0;
            sum2 += p11_lo;
            carry2 += (sum2 < p11_lo) ? 1 : 0;
            sum2 += p20_lo;
            carry2 += (sum2 < p20_lo) ? 1 : 0;

            result.hi = sum2;

            return result;
        }

        DYN_FUNC uint192_t operator/(uint64_t other) const
        {
            uint64_t rem = 0;
            uint64_t q_hi = 0, q_mid = 0, q_lo = 0;
            auto rc = dyno::precompute_div64_reciprocal(other);

            {
                uint64_t q = 0, r = 0;
                dyno::div128by64_precomputed(rem, hi, rc, q, r);
                q_hi = q;
                rem = r;
            }
            {
                uint64_t q = 0, r = 0;
                dyno::div128by64_precomputed(rem, mi, rc, q, r);
                q_mid = q;
                rem = r;
            }
            {
                uint64_t q = 0, r = 0;
                dyno::div128by64_precomputed(rem, lo, rc, q, r);
                q_lo = q;
                rem = r;
            }

            uint192_t Q{q_lo, q_mid, q_hi}; // constructor expects (lo, mid, hi)
            return Q;
        }

        DYN_FUNC uint192_t operator<<(int shift) const
        {
            if (shift == 0)
            {
                return *this;
            }
            if (shift < 64)
            {
                uint64_t new_hi = (hi << shift) | (mi >> (64 - shift));
                uint64_t new_mi = (mi << shift) | (lo >> (64 - shift));
                uint64_t new_lo = lo << shift;
                return {new_lo, new_mi, new_hi};
            }
            if (shift < 128)
            {
                uint64_t new_hi = (mi << (shift - 64)) | (lo >> (128 - shift));
                uint64_t new_mi = lo << (shift - 64);
                uint64_t new_lo = 0;
                return {new_lo, new_mi, new_hi};
            }
            if (shift < 192)
            {
                uint64_t new_hi = lo << (shift - 128);
                uint64_t new_mi = 0;
                uint64_t new_lo = 0;
                return {new_lo, new_mi, new_hi};
            }
            return {0, 0, 0};
        }
        DYN_FUNC void left_shift_with_overflow(int shift, uint64_t& u3, uint64_t& u2, uint64_t& u1, uint64_t& u0) const
        {
            assert(shift >= 0 && shift < 64);
            if (shift == 0)
            {
                u0 = lo;
                u1 = mi;
                u2 = hi;
                u3 = 0;
                return;
            }
            u0 = lo << shift;
            u1 = (lo >> (64 - shift)) | (mi << shift);
            u2 = (mi >> (64 - shift)) | (hi << shift);
            u3 = hi >> (64 - shift);
        }
        DYN_FUNC uint192_t& operator-=(const uint192_t& other)
        {
            uint64_t borrow_1 = (lo > ~other.lo) ? 1 : 0;
            lo -= other.lo;
            uint64_t borrow_2 = (mi < other.mi || (borrow_1 == 1 && mi == other.mi)) ? 1 : 0;
            mi -= other.mi + borrow_1;
            hi -= other.hi + borrow_2;
            return *this;
        }
        DYN_FUNC uint192_t operator/(const uint128_t& other) const
        {
            if (other.hi == 0)
            {
                if (hi)
                    return (*this) / other.lo;
                uint128_t lower{lo, mi};
                uint128_t result = lower / uint128_t(other.lo, other.hi);
                return {result.lo, result.hi, 0};
            }
            int s = count_leading_zeros(other.hi); // 0 <= s < 64

            uint128_t vn = other << s;

            uint64_t un3, un2, un1, un0;
            left_shift_with_overflow(s, un3, un2, un1, un0);

            uint64_t vn1 = vn.hi;
            uint64_t vn0 = vn.lo;

            // j = 1 iter
            // ---------- Step D3: estimate qhat ----------
            uint64_t qhat = estimate_qhat(un3, un2, un1, vn1, vn0);
            // ---------- Step D4: compute q̂*V ----------
            uint192_t qV = uint192_t(qhat, 0, 0) * uint192_t(vn0, vn1, 0);
            // compare qV with (un3, un2, un1, un0)
            if (cmp_3_limbs(un3, un2, un1, qV.hi, qV.mi, qV.lo) < 0)
            {
                qhat--;
                qV -= uint192_t(vn0, vn1, 0);
            }
            uint64_t qmi = qhat;
            // subtract (un3, un2, un1) with qV, but next iter we need un2, un1, un0 so un3 is discarded
            uint64_t borrow_1 = (un1 < qV.lo) ? 1 : 0;
            un1 -= qV.lo;
            un2 -= qV.mi + borrow_1;

            // j = 0 iter
            qhat = estimate_qhat(un2, un1, un0, vn1, vn0);
            qV = uint192_t(qhat, 0, 0) * uint192_t(vn0, vn1, 0);
            if (cmp_3_limbs(un2, un1, un0, qV.hi, qV.mi, qV.lo) < 0)
            {
                qhat--;
            }
            // ---------- Step D8: done ----------
            return {qhat, qmi, 0};
        }


        static DYN_FUNC bool cmp4(uint64_t a3, uint64_t a2, uint64_t a1, uint64_t a0, uint64_t b3, uint64_t b2, uint64_t b1, uint64_t b0)
        {
            if (a3 != b3) return a3 > b3;
            if (a2 != b2) return a2 > b2;
            if (a1 != b1) return a1 > b1;
            return a0 >= b0;
        }
        static DYN_FUNC void mul64x192_to_256(uint64_t a, const uint192_t& b, uint64_t& lo0, uint64_t& lo1, uint64_t& hi0, uint64_t& hi1)
        {
            uint64_t b0 = b.lo;
            uint64_t b1 = b.mi;
            uint64_t b2 = b.hi;

            uint64_t p0_lo, p0_hi;
            dyno::mul64x64_to_128(a, b0, p0_lo, p0_hi);

            uint64_t p1_lo, p1_hi;
            dyno::mul64x64_to_128(a, b1, p1_lo, p1_hi);

            uint64_t p2_lo, p2_hi;
            dyno::mul64x64_to_128(a, b2, p2_lo, p2_hi);

            lo0 = p0_lo;

            uint64_t sum1 = p0_hi + p1_lo;
            uint64_t carry1 = (sum1 < p0_hi) ? 1 : 0;

            lo1 = sum1;

            uint64_t sum2 = carry1 + p1_hi + p2_lo;
            uint64_t carry2 = (sum2 < carry1) ? 1 : 0;

            hi0 = sum2;

            hi1 = p2_hi + carry2;
        }

        DYN_FUNC uint192_t operator/(const uint192_t& other) const
        {
            if (hi < other.hi)
            {
                return {0, 0, 0};
            }
            if (other.hi == 0)
            {
                if (hi)
                    return (*this) / uint128_t(other.lo, other.mi);
                uint128_t result = uint128_t(lo, mi) / uint128_t(other.lo, other.mi);
                return {result.lo, result.hi, 0};
            }
            int s = count_leading_zeros(other.hi); // 0 <= s < 64

            uint192_t vn = other << s;

            uint64_t un3, un2, un1, un0;
            left_shift_with_overflow(s, un3, un2, un1, un0);

            uint64_t vn2 = vn.hi;
            uint64_t vn1 = vn.mi;
            uint64_t vn0 = vn.lo;

            // j = 0 iter
            // ---------- Step D3: estimate qhat ----------
            uint64_t qhat = estimate_qhat(un3, un2, un1, vn2, vn1);
            // ---------- Step D4: if q̂*V > un, dec qhat ----------
            uint64_t qv_3, qv_2, qv_1, qv_0;
            mul64x192_to_256(qhat, vn, qv_0, qv_1, qv_2, qv_3);
            // compare qV with (un3, un2, un1, un0)
            if (!cmp4(un3, un2, un1, un0, qv_3, qv_2, qv_1, qv_0))
            {
                qhat--;
            }
            // ---------- Step D8: done ----------
            return {qhat, 0, 0};
        }

        DYN_FUNC bool operator>=(const uint192_t& other) const
        {
            if (hi != other.hi) return hi > other.hi;
            if (mi != other.mi) return mi > other.mi;
            return lo >= other.lo;
        }

        DYN_FUNC bool operator<(const uint192_t& other) const
        {
            return !(*this >= other);
        }

        DYN_FUNC bool operator==(const uint192_t& other) const
        {
            return (hi == other.hi) && (mi == other.mi) && (lo == other.lo);
        }
    };

    struct uint256_t
    {
        uint64_t lo0;
        uint64_t lo1;
        uint64_t hi0;
        uint64_t hi1;

        DYN_FUNC uint256_t() : lo0(0), lo1(0), hi0(0), hi1(0)
        {
        }

        DYN_FUNC uint256_t(uint64_t _lo0, uint64_t _lo1, uint64_t _hi0, uint64_t _hi1)
            : lo0(_lo0), lo1(_lo1), hi0(_hi0), hi1(_hi1)
        {
        }

        DYN_FUNC uint256_t operator*(const uint256_t& other) const
        {
            uint64_t r0, r1, r2, r3;

            uint64_t p00_lo, p00_hi, p01_lo, p01_hi, p02_lo, p02_hi, p03_lo, p03_hi;
            uint64_t p10_lo, p10_hi, p11_lo, p11_hi, p12_lo, p12_hi, p13_lo, p13_hi;
            uint64_t p20_lo, p20_hi, p21_lo, p21_hi, p22_lo, p22_hi, p23_lo, p23_hi;
            uint64_t p30_lo, p30_hi, p31_lo, p31_hi, p32_lo, p32_hi, p33_lo, p33_hi;

            dyno::mul64x64_to_128(lo0, other.lo0, p00_lo, p00_hi);
            dyno::mul64x64_to_128(lo0, other.lo1, p01_lo, p01_hi);
            dyno::mul64x64_to_128(lo0, other.hi0, p02_lo, p02_hi);
            dyno::mul64x64_to_128(lo0, other.hi1, p03_lo, p03_hi);

            dyno::mul64x64_to_128(lo1, other.lo0, p10_lo, p10_hi);
            dyno::mul64x64_to_128(lo1, other.lo1, p11_lo, p11_hi);
            dyno::mul64x64_to_128(lo1, other.hi0, p12_lo, p12_hi);
            dyno::mul64x64_to_128(lo1, other.hi1, p13_lo, p13_hi);

            dyno::mul64x64_to_128(hi0, other.lo0, p20_lo, p20_hi);
            dyno::mul64x64_to_128(hi0, other.lo1, p21_lo, p21_hi);
            dyno::mul64x64_to_128(hi0, other.hi0, p22_lo, p22_hi);
            dyno::mul64x64_to_128(hi0, other.hi1, p23_lo, p23_hi);

            dyno::mul64x64_to_128(hi1, other.lo0, p30_lo, p30_hi);
            dyno::mul64x64_to_128(hi1, other.lo1, p31_lo, p31_hi);
            dyno::mul64x64_to_128(hi1, other.hi0, p32_lo, p32_hi);
            dyno::mul64x64_to_128(hi1, other.hi1, p33_lo, p33_hi);

            auto add64 = [](uint64_t& acc, uint64_t v, uint64_t& carry)
            {
                uint64_t tmp = acc + v;
                carry += (tmp < acc);
                acc = tmp;
            };

            r0 = p00_lo;

            uint64_t acc, carry;
            acc = p00_hi;
            carry = 0;
            add64(acc, p01_lo, carry);
            add64(acc, p10_lo, carry);
            r1 = acc;

            {
                uint64_t acc_hi = p01_hi;
                uint64_t carry_hi = 0;
                add64(acc_hi, p10_hi, carry_hi);
                add64(acc_hi, carry, carry_hi);

                uint64_t acc_lo = acc_hi;
                uint64_t carry_lo = 0;
                add64(acc_lo, p02_lo, carry_lo);
                add64(acc_lo, p11_lo, carry_lo);
                add64(acc_lo, p20_lo, carry_lo);

                r2 = acc_lo;
                carry = carry_hi + carry_lo;
            }

            {
                uint64_t acc_hi = p02_hi;
                uint64_t carry_hi = 0;
                add64(acc_hi, p11_hi, carry_hi);
                add64(acc_hi, p20_hi, carry_hi);
                add64(acc_hi, carry, carry_hi);

                uint64_t acc_lo = acc_hi;
                uint64_t carry_lo = 0;
                add64(acc_lo, p03_lo, carry_lo);
                add64(acc_lo, p12_lo, carry_lo);
                add64(acc_lo, p21_lo, carry_lo);
                add64(acc_lo, p30_lo, carry_lo);

                r3 = acc_lo;
            }

            return {r0, r1, r2, r3};
        }

        DYN_FUNC bool try_add(const uint256_t& other, uint256_t& result) const
        {
            uint64_t carry = 0;
            result.lo0 = lo0 + other.lo0;
            carry = (result.lo0 < lo0) ? 1 : 0;

            result.lo1 = lo1 + other.lo1 + carry;
            carry = (result.lo1 < lo1 || (carry == 1 && result.lo1 == lo1)) ? 1 : 0;

            result.hi0 = hi0 + other.hi0 + carry;
            carry = (result.hi0 < hi0 || (carry == 1 && result.hi0 == hi0)) ? 1 : 0;

            if (addition_will_overflow(hi1, other.hi1, carry))
                return false;
            result.hi1 = hi1 + other.hi1 + carry;
            return true;
        }

        DYN_FUNC bool try_sub(const uint256_t& other, uint256_t& result) const
        {
            if (*this < other)
                return false;
            result = *this - other;
            return true;
        }

        DYN_FUNC bool try_mul(const uint256_t& other, uint256_t& result) const
        {
            uint64_t r0, r1, r2, r3;

            uint64_t p00_lo, p00_hi, p01_lo, p01_hi, p02_lo, p02_hi, p03_lo, p03_hi;
            uint64_t p10_lo, p10_hi, p11_lo, p11_hi, p12_lo, p12_hi, p13_lo, p13_hi;
            uint64_t p20_lo, p20_hi, p21_lo, p21_hi, p22_lo, p22_hi, p23_lo, p23_hi;
            uint64_t p30_lo, p30_hi, p31_lo, p31_hi, p32_lo, p32_hi, p33_lo, p33_hi;

            dyno::mul64x64_to_128(lo0, other.lo0, p00_lo, p00_hi);
            dyno::mul64x64_to_128(lo0, other.lo1, p01_lo, p01_hi);
            dyno::mul64x64_to_128(lo0, other.hi0, p02_lo, p02_hi);
            dyno::mul64x64_to_128(lo0, other.hi1, p03_lo, p03_hi);

            dyno::mul64x64_to_128(lo1, other.lo0, p10_lo, p10_hi);
            dyno::mul64x64_to_128(lo1, other.lo1, p11_lo, p11_hi);
            dyno::mul64x64_to_128(lo1, other.hi0, p12_lo, p12_hi);
            dyno::mul64x64_to_128(lo1, other.hi1, p13_lo, p13_hi);

            dyno::mul64x64_to_128(hi0, other.lo0, p20_lo, p20_hi);
            dyno::mul64x64_to_128(hi0, other.lo1, p21_lo, p21_hi);
            dyno::mul64x64_to_128(hi0, other.hi0, p22_lo, p22_hi);
            dyno::mul64x64_to_128(hi0, other.hi1, p23_lo, p23_hi);

            dyno::mul64x64_to_128(hi1, other.lo0, p30_lo, p30_hi);
            dyno::mul64x64_to_128(hi1, other.lo1, p31_lo, p31_hi);
            dyno::mul64x64_to_128(hi1, other.hi0, p32_lo, p32_hi);
            dyno::mul64x64_to_128(hi1, other.hi1, p33_lo, p33_hi);

            auto add64 = [](uint64_t& acc, uint64_t v, uint64_t& carry)
            {
                uint64_t tmp = acc + v;
                carry += (tmp < acc);
                acc = tmp;
            };

            r0 = p00_lo;

            uint64_t acc, carry;
            acc = p00_hi;
            carry = 0;
            add64(acc, p01_lo, carry);
            add64(acc, p10_lo, carry);
            r1 = acc;

            {
                uint64_t acc_hi = p01_hi;
                uint64_t carry_hi = 0;
                add64(acc_hi, p10_hi, carry_hi);
                add64(acc_hi, carry, carry_hi);

                uint64_t acc_lo = acc_hi;
                uint64_t carry_lo = 0;
                add64(acc_lo, p02_lo, carry_lo);
                add64(acc_lo, p11_lo, carry_lo);
                add64(acc_lo, p20_lo, carry_lo);

                r2 = acc_lo;
                carry = carry_hi + carry_lo;
            }

            {
                uint64_t acc_hi = p02_hi;
                uint64_t carry_hi = 0;
                add64(acc_hi, p11_hi, carry_hi);
                if (carry_hi) return false;
                add64(acc_hi, p20_hi, carry_hi);
                if (carry_hi) return false;
                add64(acc_hi, carry, carry_hi);
                if (carry_hi) return false;
                uint64_t acc_lo = acc_hi;
                uint64_t carry_lo = 0;
                add64(acc_lo, p03_lo, carry_lo);
                if (carry_lo) return false;
                add64(acc_lo, p12_lo, carry_lo);
                if (carry_lo) return false;
                add64(acc_lo, p21_lo, carry_lo);
                if (carry_lo) return false;
                add64(acc_lo, p30_lo, carry_lo);
                if (carry_lo) return false;
                r3 = acc_lo;
            }

            result = {r0, r1, r2, r3};
            return true;
        }

        DYN_FUNC void left_shift_with_overflow(int shift, uint64_t& u4, uint64_t& u3, uint64_t& u2, uint64_t& u1, uint64_t& u0) const
        {
            assert(shift >= 0 && shift < 64);
            u0 = lo0 << shift;
            u1 = (lo0 >> (64 - shift)) | (lo1 << shift);
            u2 = (lo1 >> (64 - shift)) | (hi0 << shift);
            u3 = (hi0 >> (64 - shift)) | (hi1 << shift);
            u4 = hi1 >> (64 - shift);
        }
        DYN_FUNC uint256_t operator/(uint64_t other) const
        {
            uint256_t res;

            auto rc = dyno::precompute_div64_reciprocal(other);

            // Divide highest 64 bits
            res.hi1 = hi1 / other;
            uint64_t rem1 = hi1 % other;

            // Divide (rem1:hi0) by other
            uint64_t rem2;
            dyno::div128by64_precomputed(rem1, hi0, rc, res.hi0, rem2);

            // Divide (rem2:lo1) by other
            uint64_t rem3;
            dyno::div128by64_precomputed(rem2, lo1, rc, res.lo1, rem3);

            // Divide (rem3:lo0) by other
            dyno::div128by64_precomputed(rem3, lo0, rc, res.lo0);

            return res;
        }
        DYN_FUNC void shift_left_with_overflow(int shift, uint64_t& u4, uint64_t& u3, uint64_t& u2, uint64_t& u1, uint64_t& u0) const
        {
            assert(shift >= 0 && shift < 64);
            if (shift ==0)
            {
                u4 = 0;
                u3 = hi1;
                u2 = hi0;
                u1 = lo1;
                u0 = lo0;
                return;
            }
            u4 = hi1 >> (64 - shift);
            u3 = (hi0 >> (64 - shift)) | (hi1 << shift);
            u2 = (lo1 >> (64 - shift)) | (hi0 << shift);
            u1 = (lo0 >> (64 - shift)) | (lo1 << shift);
            u0 = lo0 << shift;
        }
        DYN_FUNC uint256_t operator/(const uint128_t& other) const
        {
            if (other.hi == 0)
            {
                if (hi1)
                    return (*this) / other.lo;
                uint192_t res = uint192_t(lo0, lo1, hi0) / uint128_t(other.lo, other.hi);
                return uint256_t(res.lo, res.mi, res.hi, 0);
            }
            int s = count_leading_zeros(other.hi); // 0 <= s < 64
            uint128_t vn = other << s;
            uint64_t un4, un3, un2, un1, un0;
            shift_left_with_overflow(s, un4, un3, un2, un1, un0);
            uint64_t vn1 = vn.hi;
            uint64_t vn0 = vn.lo;
            // j = 2 iter
            // ---------- Step D3: estimate qhat ----------
            uint64_t qhat = estimate_qhat(un4, un3, un2, vn1, vn0);
            // ---------- Step D4: compute q̂*V ----------
            uint192_t qV2 = uint192_t(qhat, 0, 0) * uint192_t(vn0, vn1, 0);
            // compare qV with (un4, un3, un2)
            if (cmp_3_limbs(un4, un3, un2, qV2.hi, qV2.mi, qV2.lo) < 0)
            {
                qhat--;
                qV2 -= uint192_t(vn0, vn1, 0);
            }
            // subtract (un4, un3, un2) with qV, but next iter we need un3, un2, un1 so un4 is discarded
            uint64_t borrow_1 = (un2 < qV2.lo) ? 1 : 0;
            un2 -= qV2.lo;
            un3 -= qV2.mi + borrow_1;
            uint64_t qhi = qhat;
            // j = 1 iter
            qhat = estimate_qhat(un3, un2, un1, vn1, vn0);
            uint192_t qV1 = uint192_t(qhat, 0, 0) * uint192_t(vn0, vn1, 0);
            if (cmp_3_limbs(un3, un2, un1, qV1.hi, qV1.mi, qV1.lo) < 0)
            {
                qhat--;
                qV1 -= uint192_t(vn0, vn1, 0);
            }
            uint64_t qmi = qhat;
            // subtract (un3, un2, un1) with qV, but next iter we need un2, un1, un0 so un3 is discarded
            borrow_1 = (un1 < qV1.lo) ? 1 : 0;
            un1 -= qV1.lo;
            un2 -= qV1.mi + borrow_1;
            // j = 0 iter
            qhat = estimate_qhat(un2, un1, un0, vn1, vn0);
            uint192_t qV0 = uint192_t(qhat, 0, 0) * uint192_t(vn0, vn1, 0);
            if (cmp_3_limbs(un2, un1, un0, qV0.hi, qV0.mi, qV0.lo) < 0)
            {
                qhat--;
            }
            // ---------- Step D8: done ----------
            return {qhat, qmi, qhi, 0};
        }

        DYN_FUNC uint256_t& operator-=(const uint256_t& other)
        {
            uint64_t borrow_1 = (lo0 < other.lo0) ? 1 : 0;
            lo0 -= other.lo0;
            uint64_t borrow_2 = (lo1 < other.lo1 || (borrow_1 == 1 && lo1 == other.lo1)) ? 1 : 0;
            lo1 -= other.lo1 + borrow_1;
            borrow_1 = (hi0 < other.hi0 || (borrow_2 == 1 && hi0 == other.hi0)) ? 1 : 0;
            hi0 -= other.hi0 + borrow_2;
            hi1 -= other.hi1 + borrow_1;
            return *this;
        }

        DYN_FUNC uint256_t operator/(const uint192_t& other) const
        {
            if (other.hi == 0)
            {
                if (hi1)
                    return (*this) / uint128_t(other.lo, other.mi);
                uint192_t result = uint192_t(lo0, lo1, hi0) / uint128_t(other.lo, other.mi);
                return {result.lo, result.mi, result.hi, 0};
            }
            int s = count_leading_zeros(other.hi); // 0 <= s < 64
            uint192_t vn = other << s;
            uint64_t un4, un3, un2, un1, un0;
            shift_left_with_overflow(s, un4, un3, un2, un1, un0);
            uint64_t vn2 = vn.hi;
            uint64_t vn1 = vn.mi;
            uint64_t vn0 = vn.lo;
            // j = 1 iter
            // ---------- Step D3: estimate qhat ----------
            uint64_t qhat = estimate_qhat(un4, un3, un2, vn2, vn1);
            // ---------- Step D4: compute q̂*V ----------
            uint256_t qV;
            uint192_t::mul64x192_to_256(qhat, vn, qV.lo0, qV.lo1, qV.hi0, qV.hi1);
            // compare qV with (un4, un3, un2, un1)
            if (!greater_than_4_limbs(un4, un3, un2, un1, qV.hi1, qV.hi0, qV.lo1, qV.lo0))
            {
                qhat--;
                qV -= uint256_t(vn0, vn1, vn2, 0);
            }
            // subtract (un4, un3, un2, un1) with qV, but next iter we need un3, un2, un1 so un4 is discarded
            uint64_t borrow_1 = (un1 < qV.lo0) ? 1 : 0;
            un1 -= qV.lo0;
            uint64_t borrow_2 = (un2 < qV.lo1 || (borrow_1 == 1 && un2 == qV.lo1)) ? 1 : 0;
            un2 -= qV.lo1 + borrow_1;
            un3 -= qV.hi0 + borrow_2;
            uint64_t qhi = qhat;
            // j = 0 iter
            qhat = estimate_qhat(un3, un2, un1, vn2, vn1);
            uint256_t qV0;
            uint192_t::mul64x192_to_256(qhat, vn, qV0.lo0, qV0.lo1, qV0.hi0, qV0.hi1);
            if (!greater_than_4_limbs(un3, un2, un1, un0, qV0.hi1, qV0.hi0, qV0.lo1, qV0.lo0))
            {
                qhat--;
            }
            // ---------- Step D8: done ----------
            return {qhat, qhi, 0, 0};
        }
        DYN_FUNC uint256_t operator<<(int shift) const
        {
            assert(shift >= 0 && shift < 256);
            if (shift == 0)
            {
                return *this;
            }
            if (shift < 64)
            {
                return {
                    lo0 << shift,
                    (lo0 >> (64 - shift)) | (lo1 << shift),
                    (lo1 >> (64 - shift)) | (hi0 << shift),
                    (hi0 >> (64 - shift)) | (hi1 << shift)};
            }
            if (shift < 128)
            {
                shift -= 64;
                return {
                    0,
                    lo0 << shift,
                    (lo0 >> (64 - shift)) | (lo1 << shift),
                    (lo1 >> (64 - shift)) | (hi0 << shift)};
            }
            if (shift < 192)
            {
                shift -= 128;
                return {
                    0,
                    0,
                    lo0 << shift,
                    (lo0 >> (64 - shift)) | (lo1 << shift)};
            }
             // if (shift < 256)
            {
                shift -= 192;
                return {
                    0,
                    0,
                    0,
                    lo0 << shift};
            }
        }
        DYN_FUNC static void mul64x256_to_320(uint64_t a, const uint256_t& b,
                                     uint64_t& p0, uint64_t& p1, uint64_t& p2, uint64_t& p3, uint64_t& p4)
        {
            uint128_t t;
            uint64_t carry;

            t = uint128_t::from_uint64_mul(a, b.lo0);
            p0 = t.lo;
            carry = t.hi;

            t = uint128_t::from_uint64_mul(a, b.lo1);
            uint64_t sum1 = t.lo + carry;
            uint64_t carry1 = (sum1 < t.lo) ? 1 : 0;
            p1 = sum1;
            carry = t.hi + carry1;

            t = uint128_t::from_uint64_mul(a, b.hi0);
            uint64_t sum2 = t.lo + carry;
            uint64_t carry2 = (sum2 < t.lo) ? 1 : 0;
            p2 = sum2;
            carry = t.hi + carry2;

            t = uint128_t::from_uint64_mul(a, b.hi1);
            uint64_t sum3 = t.lo + carry;
            uint64_t carry3 = (sum3 < t.lo) ? 1 : 0;
            p3 = sum3;
            p4 = t.hi + carry3;
        }

        DYN_FUNC uint256_t operator/(const uint256_t& other) const
        {
            if (hi1 < other.hi1)
            {
                return {0, 0, 0, 0};
            }
            if (other.hi1 == 0)
            {
                if (hi1)
                    return (*this) / uint192_t{other.lo0, other.lo1, other.hi0};
                uint192_t result;
                result = uint192_t(lo0, lo1, hi0) / uint192_t(other.lo0, other.lo1, other.hi0);
                return {result.lo, result.mi, result.hi, 0};
            }
            int s = count_leading_zeros(other.hi1); // 0 <= s < 64
            uint64_t un4, un3, un2, un1, un0;
            uint256_t vn = other << s;
            shift_left_with_overflow(s, un4, un3, un2, un1, un0);
            uint64_t vn3 = vn.hi1;
            uint64_t vn2 = vn.hi0;
            uint64_t vn1 = vn.lo1;
            uint64_t vn0 = vn.lo0;
            // j = 0 iter
            // ---------- Step D3: estimate qhat ----------
            uint64_t qhat = estimate_qhat(un4, un3, un2, vn3, vn2);
            // ---------- Step D4: compute q̂*V ----------
            uint64_t qV_0, qV_1, qV_2, qV_3, qV_4;
            mul64x256_to_320(qhat, vn, qV_0, qV_1, qV_2, qV_3, qV_4);
            // compare qV with (un4, un3, un2, un1, un0)
            if (!cmp_5_limbs(un4, un3, un2, un1, un0, qV_4, qV_3, qV_2, qV_1, qV_0))
            {
                qhat--;
            }
            return {qhat, 0, 0, 0};
        }


        DYN_FUNC uint256_t operator+(const uint256_t& other) const
        {
            uint256_t result;
            result.lo0 = lo0 + other.lo0;
            uint64_t c = (result.lo0 < lo0) ? 1 : 0;
            result.lo1 = lo1 + other.lo1 + c;
            c = (result.lo1 < lo1 || (c == 1 && result.lo1 == lo1)) ? 1 : 0;
            result.hi0 = hi0 + other.hi0 + c;
            c = (result.hi0 < hi0 || (c == 1 && result.hi0 == hi0)) ? 1 : 0;
            result.hi1 = hi1 + other.hi1 + c;
            return result;
        }

        DYN_FUNC uint256_t operator-(const uint256_t& other) const
        {
            uint256_t result;
            result.lo0 = lo0 - other.lo0;
            uint64_t c = (lo0 < other.lo0) ? 1 : 0;
            result.lo1 = lo1 - other.lo1 - c;
            c = ((lo1 < other.lo1) || (c && lo1 == other.lo1)) ? 1 : 0;
            result.hi0 = hi0 - other.hi0 - c;
            c = ((hi0 < other.hi0) || (c && hi0 == other.hi0)) ? 1 : 0;
            result.hi1 = hi1 - other.hi1 - c;
            return result;
        }

        bool operator<(const uint256_t& uint256) const
        {
            return (hi1 < uint256.hi1) || (hi1 == uint256.hi1 && hi0 < uint256.hi0) ||
                (hi1 == uint256.hi1 && hi0 == uint256.hi0 && lo1 < uint256.lo1) ||
                (hi1 == uint256.hi1 && hi0 == uint256.hi0 && lo1 == uint256.lo1 && lo0 < uint256.lo0);
        }

        bool operator==(const uint256_t& uint256) const
        {
            return (lo0 == uint256.lo0) && (lo1 == uint256.lo1) &&
                   (hi0 == uint256.hi0) && (hi1 == uint256.hi1);
        }
    };

    struct ext_sgn_int192_t
    {
        uint64_t limb0;
        uint64_t limb1;
        uint64_t limb2;
        uint64_t neg_mask;

        DYN_FUNC ext_sgn_int192_t() : limb0(0), limb1(0), limb2(0), neg_mask(0)
        {
        }

        DYN_FUNC ext_sgn_int192_t(uint64_t l0, uint64_t l1, uint64_t l2, bool negative)
            : limb0(l0), limb1(l1), limb2(l2), neg_mask(negative ? ~0ULL : 0)
        {
        }

        DYN_FUNC static ext_sgn_int192_t from_int128_int64_mul(const ext_sgn_int128_t& a, int64_t b)
        {
            bool negative = (a.neg_mask != 0) ^ (b < 0);

            uint64_t ua = a.lo;
            uint64_t ub = a.hi;
            uint64_t uc = (b < 0) ? static_cast<uint64_t>(-b) : static_cast<uint64_t>(b);

            uint64_t lo_lo, lo_hi;
            dyno::mul64x64_to_128(ua, uc, lo_lo, lo_hi);

            uint64_t hi_lo, hi_hi;
            dyno::mul64x64_to_128(ub, uc, hi_lo, hi_hi);

            uint64_t limb0 = lo_lo;

            uint64_t limb1 = lo_hi + hi_lo;
            uint64_t carry = 0;
            if (limb1 < lo_hi) carry = 1;

            uint64_t limb2 = hi_hi + carry;

            return {limb0, limb1, limb2, negative};
        }
        DYN_FUNC static void add192(uint64_t a0, uint64_t a1, uint64_t a2,
                                    uint64_t b0, uint64_t b1, uint64_t b2,
                                    uint64_t& res0, uint64_t& res1, uint64_t& res2)
        {
            res0 = a0 + b0;
            uint64_t c0 = res0 < a0;
            res1 = a1 + b1 + c0;
            uint64_t c1 = (res1 < a1) || (c0 && res1 == a1);
            res2 = a2 + b2 + c1;
        }

        DYN_FUNC static void sub192(uint64_t a0, uint64_t a1, uint64_t a2,
                                    uint64_t b0, uint64_t b1, uint64_t b2,
                                    uint64_t& res0, uint64_t& res1, uint64_t& res2)
        {
            res0 = a0 - b0;
            uint64_t borrow0 = a0 < b0;
            res1 = a1 - b1 - borrow0;
            uint64_t borrow1 = (a1 < b1) || (borrow0 && a1 == b1);
            res2 = a2 - b2 - borrow1;
        }

        DYN_FUNC [[nodiscard]] int sgn() const
        {
            if (limb0 == 0 && limb1 == 0 && limb2 == 0)
                return 0;
            return (neg_mask != 0) ? -1 : 1;
        }
        DYN_FUNC bool try_add(const ext_sgn_int192_t& b, ext_sgn_int192_t& result) const
        {
            uint192_t abs_a = uint192_t(limb0, limb1, limb2);
            uint192_t abs_b = uint192_t(limb0, limb1, limb2);
            uint192_t abs_res;
            if (neg_mask == b.neg_mask)
            {
                if (!abs_a.try_add(abs_b, abs_res))
                    return false;
                result.limb0 = abs_res.lo;
                result.limb1 = abs_res.mi;
                result.limb2 = abs_res.hi;
                result.neg_mask = neg_mask;
                return true;
            }
            if (!abs_a.try_sub(abs_b, abs_res))
                return false;
            result.limb0 = abs_res.lo;
            result.limb1 = abs_res.mi;
            result.limb2 = abs_res.hi;
            uint64_t a_ge_b = dyno::abs_greater_mask(limb2, limb1, limb0, b.limb2, b.limb1, b.limb0);
            result.neg_mask = (a_ge_b & neg_mask) | (~a_ge_b & b.neg_mask);
            return true;
        }
        DYN_FUNC bool try_sub(const ext_sgn_int192_t& b, ext_sgn_int192_t& result) const
        {
            ext_sgn_int192_t neg_b;
            neg_b.limb0 = b.limb0;
            neg_b.limb1 = b.limb1;
            neg_b.limb2 = b.limb2;
            neg_b.neg_mask = ~b.neg_mask;
            return try_add(neg_b, result);
        }
        DYN_FUNC bool try_mul(const ext_sgn_int192_t& b, ext_sgn_int192_t& res) const
        {
            uint192_t abs_a = uint192_t(limb0, limb1, limb2);
            uint192_t abs_b = uint192_t(limb0, limb1, limb2);
            uint192_t abs_res;
            if (!abs_a.try_mul(abs_b, abs_res))
                return false;
            res.neg_mask = neg_mask ^ b.neg_mask;
            res.limb0 = abs_res.lo;
            res.limb1 = abs_res.mi;
            res.limb2 = abs_res.hi;
            return true;
        }
        DYN_FUNC ext_sgn_int192_t operator+(const ext_sgn_int192_t& b) const
        {
            uint64_t same_sign = ~(neg_mask ^ b.neg_mask);
            uint64_t diff_sign = neg_mask ^ b.neg_mask;

            uint64_t sum0, sum1, sum2;
            add192(limb0, limb1, limb2, b.limb0, b.limb1, b.limb2, sum0, sum1, sum2);

            uint64_t a_ge_b = dyno::abs_greater_mask(limb2, limb1, limb0, b.limb2, b.limb1, b.limb0);
            uint64_t d0, d1, d2;
            uint64_t hi = (a_ge_b & limb2) | (~a_ge_b & b.limb2);
            uint64_t mi = (a_ge_b & limb1) | (~a_ge_b & b.limb1);
            uint64_t lo = (a_ge_b & limb0) | (~a_ge_b & b.limb0);
            uint64_t other_hi = (a_ge_b & b.limb2) | (~a_ge_b & limb2);
            uint64_t other_mi = (a_ge_b & b.limb1) | (~a_ge_b & limb1);
            uint64_t other_lo = (a_ge_b & b.limb0) | (~a_ge_b & limb0);
            sub192(lo, mi, hi, other_lo, other_mi, other_hi, d0, d1, d2);

            uint64_t res0 = (sum0 & same_sign) | (d0 & diff_sign);
            uint64_t res1 = (sum1 & same_sign) | (d1 & diff_sign);
            uint64_t res2 = (sum2 & same_sign) | (d2 & diff_sign);
            uint64_t res_neg = (same_sign & neg_mask) | (diff_sign & ((a_ge_b & neg_mask) | (~a_ge_b & b.neg_mask)));

            return ext_sgn_int192_t(res0, res1, res2, res_neg != 0);
        }

        DYN_FUNC ext_sgn_int192_t operator-(const ext_sgn_int192_t& b) const
        {
            uint64_t b_neg_mask_flipped = ~b.neg_mask;

            uint64_t same_sign = ~(neg_mask ^ b_neg_mask_flipped);
            uint64_t diff_sign = neg_mask ^ b_neg_mask_flipped;

            uint64_t sum0, sum1, sum2;
            add192(limb0, limb1, limb2, b.limb0, b.limb1, b.limb2, sum0, sum1, sum2);

            uint64_t a_ge_b = dyno::abs_greater_mask(limb2, limb1, limb0, b.limb2, b.limb1, b.limb0);
            uint64_t d0, d1, d2;
            uint64_t hi = (a_ge_b & limb2) | (~a_ge_b & b.limb2);
            uint64_t mi = (a_ge_b & limb1) | (~a_ge_b & b.limb1);
            uint64_t lo = (a_ge_b & limb0) | (~a_ge_b & b.limb0);
            uint64_t other_hi = (a_ge_b & b.limb2) | (~a_ge_b & limb2);
            uint64_t other_mi = (a_ge_b & b.limb1) | (~a_ge_b & limb1);
            uint64_t other_lo = (a_ge_b & b.limb0) | (~a_ge_b & limb0);
            sub192(lo, mi, hi, other_lo, other_mi, other_hi, d0, d1, d2);

            uint64_t res0 = (sum0 & same_sign) | (d0 & diff_sign);
            uint64_t res1 = (sum1 & same_sign) | (d1 & diff_sign);
            uint64_t res2 = (sum2 & same_sign) | (d2 & diff_sign);
            uint64_t res_neg = (same_sign & neg_mask) | (diff_sign & ((a_ge_b & neg_mask) | (~a_ge_b &
                b_neg_mask_flipped)));

            return ext_sgn_int192_t(res0, res1, res2, res_neg != 0);
        }

        DYN_FUNC ext_sgn_int192_t operator*(const ext_sgn_int192_t& other) const
        {
            auto abs192 = [](const ext_sgn_int192_t& x)
            {
                if (x.neg_mask == 0)
                {
                    return x;
                }
                uint64_t l0 = ~x.limb0 + 1;
                uint64_t carry = (l0 == 0);
                uint64_t l1 = ~x.limb1 + carry;
                carry = (carry && l1 == 0);
                uint64_t l2 = ~x.limb2 + carry;
                return ext_sgn_int192_t(l0, l1, l2, false);
            };

            ext_sgn_int192_t a = abs192(*this);
            ext_sgn_int192_t b = abs192(other);

            uint64_t a0 = a.limb0, a1 = a.limb1, a2 = a.limb2;
            uint64_t b0 = b.limb0, b1 = b.limb1, b2 = b.limb2;

            uint64_t r0 = 0, r1 = 0, r2 = 0;

            uint64_t lo, hi, carry;

            dyno::mul64x64_to_128(a0, b0, lo, hi);
            r0 = lo;
            r1 = hi;

            dyno::mul64x64_to_128(a0, b1, lo, hi);
            uint64_t tmp = r1 + lo;
            carry = (tmp < r1);
            r1 = tmp;
            r2 += hi + carry;

            dyno::mul64x64_to_128(a1, b0, lo, hi);
            tmp = r1 + lo;
            carry = (tmp < r1);
            r1 = tmp;
            r2 += hi + carry;

            dyno::mul64x64_to_128(a0, b2, lo, hi);
            r2 += lo;

            dyno::mul64x64_to_128(a1, b1, lo, hi);
            r2 += lo;

            dyno::mul64x64_to_128(a2, b0, lo, hi);
            r2 += lo;


            bool negative = (this->neg_mask ^ other.neg_mask) != 0;

            ext_sgn_int192_t res(r0, r1, r2, negative);

            if (negative)
            {
                uint64_t l0 = ~res.limb0 + 1;
                uint64_t carry = (l0 == 0);
                uint64_t l1 = ~res.limb1 + carry;
                carry = (carry && l1 == 0);
                uint64_t l2 = ~res.limb2 + carry;
                res = ext_sgn_int192_t(l0, l1, l2, true);
            }

            return res;
        }

        DYN_FUNC static ext_sgn_int192_t from_int64_mul(int64_t a, int64_t b, int64_t c)
        {
            bool negative = (a < 0) ^ (b < 0) ^ (c < 0);

            auto abs64 = [](int64_t x) -> uint64_t
            {
                return static_cast<uint64_t>(x < 0 ? uint64_t(-(x + (x == INT64_MIN))) + (x == INT64_MIN) : x);
            };
            uint64_t ua = abs64(a);
            uint64_t ub = abs64(b);
            uint64_t uc = abs64(c);

            uint64_t ab_lo, ab_hi;
            dyno::mul64x64_to_128(ua, ub, ab_lo, ab_hi);

            uint64_t tmp_lo, tmp_hi;
            dyno::mul64x64_to_128(ab_lo, uc, tmp_lo, tmp_hi);

            uint64_t limb0 = tmp_lo;
            uint64_t limb1 = tmp_hi;
            uint64_t limb2 = 0;

            uint64_t hi0, hi1;
            dyno::mul64x64_to_128(ab_hi, uc, hi0, hi1);

            uint64_t carry = 0;
            limb1 += hi0;
            if (limb1 < hi0) carry = 1;
            limb2 += hi1 + carry;

            return {limb0, limb1, limb2, negative};
        }

        DYN_FUNC bool operator<(const ext_sgn_int192_t& b) const
        {
            bool zero_this = (limb0 == 0 && limb1 == 0 && limb2 == 0);
            bool zero_other = (b.limb0 == 0 && b.limb1 == 0 && b.limb2 == 0);
            uint64_t eff_neg_mask = zero_this ? 0 : neg_mask;
            uint64_t eff_b_neg_mask = zero_other ? 0 : b.neg_mask;

            if ((eff_neg_mask != 0) != (eff_b_neg_mask != 0)) return (eff_neg_mask != 0);
            bool abs_gt = dyno::abs_greater_mask(limb2, limb1, limb0, b.limb2, b.limb1, b.limb0) != 0;
            bool abs_eq = (limb2 == b.limb2) && (limb1 == b.limb1) && (limb0 == b.limb0);
            if (eff_neg_mask == 0)
            {
                return (!abs_gt) && !abs_eq;
            }
            else
            {
                return abs_gt;
            }
        }

        DYN_FUNC bool operator==(const ext_sgn_int192_t& b) const
        {
            bool zero_this = (limb0 == 0 && limb1 == 0 && limb2 == 0);
            bool zero_other = (b.limb0 == 0 && b.limb1 == 0 && b.limb2 == 0);
            if (zero_this && zero_other) return true;
            return (limb0 == b.limb0) && (limb1 == b.limb1) && (limb2 == b.limb2) && (neg_mask == b.neg_mask);
        }

        DYN_FUNC ext_sgn_int192_t operator/(const ext_sgn_int192_t& other) const
        {
            // Handle division by zero
            if (other.limb0 == 0 && other.limb1 == 0 && other.limb2 == 0)
            {
                // Return max value as error
                return ext_sgn_int192_t(~0ULL, ~0ULL, ~0ULL, true);
            }

            // Get absolute values
            uint192_t abs_this = {limb0, limb1, limb2};
            uint192_t abs_other = {other.limb0, other.limb1, other.limb2};

            // Perform unsigned division
            uint192_t abs_result = abs_this / abs_other;

            // Determine sign of result
            bool result_negative = (neg_mask ^ other.neg_mask) != 0;

            return {abs_result.lo, abs_result.mi, abs_result.hi, result_negative};
        }
    };

    struct ext_sgn_int256_t
    {
        uint64_t lo0;
        uint64_t lo1;
        uint64_t hi0;
        uint64_t hi1;
        uint64_t neg_mask;

        DYN_FUNC ext_sgn_int256_t()
            : lo0(0), lo1(0), hi0(0), hi1(0), neg_mask(0)
        {
        }

        DYN_FUNC ext_sgn_int256_t(uint64_t _lo0, uint64_t _lo1, uint64_t _hi0, uint64_t _hi1, bool negative)
            : lo0(_lo0), lo1(_lo1), hi0(_hi0), hi1(_hi1), neg_mask(negative ? ~0ULL : 0)
        {
        }

        DYN_FUNC ext_sgn_int256_t(int64_t a)
            : lo1(0), hi0(0), hi1(0)
        {
            neg_mask = (a < 0) ? ~0ULL : 0;
            lo0 = static_cast<uint64_t>(a < 0 ? uint64_t(-(a + (a == INT64_MIN))) + (a == INT64_MIN) : a);
        }

        DYN_FUNC ext_sgn_int256_t(ext_sgn_int128_t a)
            : hi0(0), hi1(0)
        {
            neg_mask = a.neg_mask;
            lo1 = a.hi;
            lo0 = a.lo;
        }

        DYN_FUNC static void add256_u(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                      uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3,
                                      uint64_t& r0, uint64_t& r1, uint64_t& r2, uint64_t& r3)
        {
            uint64_t t, c;
            r0 = a0 + b0;
            c = (r0 < a0);

            t = a1 + b1 + c;
            c = (t < a1) || (c && t == a1);
            r1 = t;

            t = a2 + b2 + c;
            c = (t < a2) || (c && t == a2);
            r2 = t;

            r3 = a3 + b3 + c;
        }

        DYN_FUNC static ext_sgn_int256_t from_s192_int64_mul(const ext_sgn_int192_t& a, int64_t b)
        {
            bool negative = (a.neg_mask != 0) ^ (b < 0);

            uint64_t p00_lo, p00_hi;
            uint64_t p10_lo, p10_hi;
            uint64_t p20_lo, p20_hi;
            uint64_t a0 = a.limb0;
            uint64_t a1 = a.limb1;
            uint64_t a2 = a.limb2;
            uint64_t b_abs = (b < 0) ? static_cast<uint64_t>(-b) : static_cast<uint64_t>(b);
            uint64_t r0, r1, r2, r3;
            dyno::mul64x64_to_128(a0, b_abs, p00_lo, p00_hi);

            dyno::mul64x64_to_128(a1, b_abs, p10_lo, p10_hi);

            dyno::mul64x64_to_128(a2, b_abs, p20_lo, p20_hi);
            r0 = p00_lo;

            {
                uint64_t acc = p00_hi;
                uint64_t tmp, carry = 0;
                tmp = acc;
                carry += (tmp < acc);
                acc = tmp;
                tmp = acc + p10_lo;
                carry += (tmp < acc);
                acc = tmp;
                r1 = acc;

                uint64_t carry_to_2 = p10_hi + carry;

                acc = carry_to_2;
                tmp = acc;
                carry = (tmp < acc);
                acc = tmp;
                tmp = acc;
                carry += (tmp < acc);
                acc = tmp;
                tmp = acc + p20_lo;
                carry += (tmp < acc);
                acc = tmp;
                r2 = acc;

                uint64_t carry_to_3 = p20_hi + carry;

                acc = carry_to_3;
                tmp = acc;
                acc = tmp;
                tmp = acc;
                acc = tmp;
                tmp = acc;
                acc = tmp;
                tmp = acc;
                acc = tmp;
                r3 = acc;
            }

            return ext_sgn_int256_t(r0, r1, r2, r3, negative);
        }
        DYN_FUNC static void sub256_u(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                      uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3,
                                      uint64_t& r0, uint64_t& r1, uint64_t& r2, uint64_t& r3)
        {
            uint64_t t, brw;
            r0 = a0 - b0;
            brw = (a0 < b0);

            t = a1 - b1 - brw;
            brw = ((a1 < b1) || (brw && a1 == b1)) ? 1 : 0;
            r1 = t;

            t = a2 - b2 - brw;
            brw = ((a2 < b2) || (brw && a2 == b2)) ? 1 : 0;
            r2 = t;

            r3 = a3 - b3 - brw;
        }

        DYN_FUNC static void mul256_u(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                      uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3,
                                      uint64_t& r0, uint64_t& r1, uint64_t& r2, uint64_t& r3)
        {
            uint64_t p00_lo, p00_hi, p01_lo, p01_hi, p02_lo, p02_hi, p03_lo, p03_hi;
            uint64_t p10_lo, p10_hi, p11_lo, p11_hi, p12_lo, p12_hi, p13_lo, p13_hi;
            uint64_t p20_lo, p20_hi, p21_lo, p21_hi, p22_lo, p22_hi, p23_lo, p23_hi;
            uint64_t p30_lo, p30_hi, p31_lo, p31_hi, p32_lo, p32_hi, p33_lo, p33_hi;

            dyno::mul64x64_to_128(a0, b0, p00_lo, p00_hi);
            dyno::mul64x64_to_128(a0, b1, p01_lo, p01_hi);
            dyno::mul64x64_to_128(a0, b2, p02_lo, p02_hi);
            dyno::mul64x64_to_128(a0, b3, p03_lo, p03_hi);

            dyno::mul64x64_to_128(a1, b0, p10_lo, p10_hi);
            dyno::mul64x64_to_128(a1, b1, p11_lo, p11_hi);
            dyno::mul64x64_to_128(a1, b2, p12_lo, p12_hi);
            dyno::mul64x64_to_128(a1, b3, p13_lo, p13_hi);

            dyno::mul64x64_to_128(a2, b0, p20_lo, p20_hi);
            dyno::mul64x64_to_128(a2, b1, p21_lo, p21_hi);
            dyno::mul64x64_to_128(a2, b2, p22_lo, p22_hi);
            dyno::mul64x64_to_128(a2, b3, p23_lo, p23_hi);

            dyno::mul64x64_to_128(a3, b0, p30_lo, p30_hi);
            dyno::mul64x64_to_128(a3, b1, p31_lo, p31_hi);
            dyno::mul64x64_to_128(a3, b2, p32_lo, p32_hi);
            dyno::mul64x64_to_128(a3, b3, p33_lo, p33_hi);

            r0 = p00_lo;

            {
                uint64_t acc = p00_hi;
                uint64_t tmp, carry = 0;
                tmp = acc + p01_lo;
                carry += (tmp < acc);
                acc = tmp;
                tmp = acc + p10_lo;
                carry += (tmp < acc);
                acc = tmp;
                r1 = acc;

                uint64_t carry_to_2 = p01_hi + p10_hi + carry;

                acc = carry_to_2;
                tmp = acc + p02_lo;
                carry = (tmp < acc);
                acc = tmp;
                tmp = acc + p11_lo;
                carry += (tmp < acc);
                acc = tmp;
                tmp = acc + p20_lo;
                carry += (tmp < acc);
                acc = tmp;
                r2 = acc;

                uint64_t carry_to_3 = p02_hi + p11_hi + p20_hi + carry;

                acc = carry_to_3;
                tmp = acc + p03_lo;
                acc = tmp;
                tmp = acc + p12_lo;
                acc = tmp;
                tmp = acc + p21_lo;
                acc = tmp;
                tmp = acc + p30_lo;
                acc = tmp;
                r3 = acc;
            }
        }

        DYN_FUNC bool operator==(const ext_sgn_int256_t& other) const
        {
            bool zero_this = (lo0 == 0 && lo1 == 0 && hi0 == 0 && hi1 == 0);
            bool zero_other = (other.lo0 == 0 && other.lo1 == 0 && other.hi0 == 0 && other.hi1 == 0);
            if (zero_this && zero_other) return true;
            return (neg_mask == other.neg_mask)
                && (lo0 == other.lo0) && (lo1 == other.lo1)
                && (hi0 == other.hi0) && (hi1 == other.hi1);
        }

        DYN_FUNC bool operator!=(const ext_sgn_int256_t& other) const { return !(*this == other); }

        DYN_FUNC bool operator<(const ext_sgn_int256_t& other) const
        {
            bool zero_this = (lo0 == 0 && lo1 == 0 && hi0 == 0 && hi1 == 0);
            bool zero_other = (other.lo0 == 0 && other.lo1 == 0 && other.hi0 == 0 && other.hi1 == 0);
            uint64_t eff_neg = zero_this ? 0 : neg_mask;
            uint64_t eff_other_neg = zero_other ? 0 : other.neg_mask;

            bool signs_differ = (eff_neg != eff_other_neg);
            if (signs_differ) return (eff_neg != 0);

            bool absLess = dyno::lt256_u(lo0, lo1, hi0, hi1, other.lo0, other.lo1, other.hi0, other.hi1);
            if (eff_neg == 0)
            {
                return absLess;
            }
            else
            {
                return dyno::abs_gt_u(lo0, lo1, hi0, hi1, other.lo0, other.lo1, other.hi0, other.hi1);
            }
        }

        DYN_FUNC bool operator>(const ext_sgn_int256_t& other) const { return other < *this; }
        DYN_FUNC bool operator<=(const ext_sgn_int256_t& other) const { return !(*this > other); }
        DYN_FUNC bool operator>=(const ext_sgn_int256_t& other) const { return !(*this < other); }

        DYN_FUNC bool try_add(const ext_sgn_int256_t& other, ext_sgn_int256_t& result) const
        {
            uint256_t abs_a = uint256_t{lo0, lo1, hi0, hi1};
            uint256_t abs_b = uint256_t{other.lo0, other.lo1, other.hi0, other.hi1};
            uint256_t abs_res;
            if (neg_mask == other.neg_mask)
            {
                if (!abs_a.try_add(abs_b, abs_res))
                    return false;
                result.lo0 = abs_res.lo0;
                result.lo1 = abs_res.lo1;
                result.hi0 = abs_res.hi0;
                result.hi1 = abs_res.hi1;
                result.neg_mask = neg_mask;
                return true;
            }
            if (!abs_a.try_sub(abs_b, abs_res))
                return false;
            result.lo0 = abs_res.lo0;
            result.lo1 = abs_res.lo1;
            result.hi0 = abs_res.hi0;
            result.hi1 = abs_res.hi1;
            uint64_t a_ge_b = dyno::abs_ge_mask_u(lo0, lo1, hi0, hi1, other.lo0, other.lo1, other.hi0, other.hi1);
            result.neg_mask = (a_ge_b & neg_mask) | (~a_ge_b & other.neg_mask);
            return true;
        }
        DYN_FUNC bool try_sub(const ext_sgn_int256_t& other, ext_sgn_int256_t& result) const
        {
            ext_sgn_int256_t neg_b = other;
            neg_b.neg_mask = ~neg_b.neg_mask;
            return try_add(neg_b, result);
        }
        DYN_FUNC bool try_mul(const ext_sgn_int256_t& other, ext_sgn_int256_t& res) const
        {
            uint256_t abs_a = uint256_t{lo0, lo1, hi0, hi1};
            uint256_t abs_b = uint256_t{other.lo0, other.lo1, other.hi0, other.hi1};
            uint256_t abs_res;
            if (!abs_a.try_mul(abs_b, abs_res))
                return false;
            res.neg_mask = neg_mask ^ other.neg_mask;
            res.lo0 = abs_res.lo0;
            res.lo1 = abs_res.lo1;
            res.hi0 = abs_res.hi0;
            res.hi1 = abs_res.hi1;
            return true;
        }
        DYN_FUNC ext_sgn_int256_t operator+(const ext_sgn_int256_t& other) const
        {
            uint64_t sum0, sum1, sum2, sum3;
            add256_u(lo0, lo1, hi0, hi1, other.lo0, other.lo1, other.hi0, other.hi1, sum0, sum1, sum2, sum3);

            uint64_t diff_ab0, diff_ab1, diff_ab2, diff_ab3;
            sub256_u(lo0, lo1, hi0, hi1, other.lo0, other.lo1, other.hi0, other.hi1, diff_ab0, diff_ab1, diff_ab2,
                     diff_ab3);

            uint64_t diff_ba0, diff_ba1, diff_ba2, diff_ba3;
            sub256_u(other.lo0, other.lo1, other.hi0, other.hi1, lo0, lo1, hi0, hi1, diff_ba0, diff_ba1, diff_ba2,
                     diff_ba3);

            uint64_t same_mask = ~(neg_mask ^ other.neg_mask);

            uint64_t agemask = dyno::abs_ge_mask_u(lo0, lo1, hi0, hi1, other.lo0, other.lo1, other.hi0, other.hi1);
            uint64_t chosen0 = (diff_ab0 & agemask) | (diff_ba0 & ~agemask);
            uint64_t chosen1 = (diff_ab1 & agemask) | (diff_ba1 & ~agemask);
            uint64_t chosen2 = (diff_ab2 & agemask) | (diff_ba2 & ~agemask);
            uint64_t chosen3 = (diff_ab3 & agemask) | (diff_ba3 & ~agemask);

            uint64_t r0 = (sum0 & same_mask) | (chosen0 & ~same_mask);
            uint64_t r1 = (sum1 & same_mask) | (chosen1 & ~same_mask);
            uint64_t r2 = (sum2 & same_mask) | (chosen2 & ~same_mask);
            uint64_t r3 = (sum3 & same_mask) | (chosen3 & ~same_mask);

            uint64_t chosen_neg = (neg_mask & agemask) | (other.neg_mask & ~agemask);
            uint64_t res_neg = (neg_mask & same_mask) | (chosen_neg & ~same_mask);

            return ext_sgn_int256_t(r0, r1, r2, r3, (res_neg != 0));
        }

        DYN_FUNC ext_sgn_int256_t operator-(const ext_sgn_int256_t& other) const
        {
            ext_sgn_int256_t nb = other;
            nb.neg_mask = ~nb.neg_mask;
            return *this + nb;
        }

        DYN_FUNC ext_sgn_int256_t operator*(const ext_sgn_int256_t& other) const
        {
            uint64_t r0, r1, r2, r3;
            mul256_u(lo0, lo1, hi0, hi1, other.lo0, other.lo1, other.hi0, other.hi1, r0, r1, r2, r3);
            uint64_t res_neg_mask = neg_mask ^ other.neg_mask;
            if (r0 == 0 && r1 == 0 && r2 == 0 && r3 == 0) res_neg_mask = 0;
            return ext_sgn_int256_t(r0, r1, r2, r3, (res_neg_mask != 0));
        }


        DYN_FUNC ext_sgn_int256_t operator/(const ext_sgn_int256_t& other) const
        {
            // Handle division by zero
            if (other.lo0 == 0 && other.lo1 == 0 && other.hi0 == 0 && other.hi1 == 0)
            {
                // Return max value as error
                return ext_sgn_int256_t(~0ULL, ~0ULL, ~0ULL, ~0ULL, true);
            }

            // Get absolute values
            uint256_t abs_this = {lo0, lo1, hi0, hi1};
            uint256_t abs_other = {other.lo0, other.lo1, other.hi0, other.hi1};

            // Perform unsigned division
            uint256_t abs_result = abs_this / abs_other;

            // Determine sign of result
            bool result_negative = (neg_mask ^ other.neg_mask) != 0;

            return ext_sgn_int256_t(abs_result.lo0, abs_result.lo1, abs_result.hi0, abs_result.hi1, result_negative);
        }
    };

    struct uint512_t
    {
        uint64_t lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7;

        DYN_FUNC uint512_t() : lo0(0), lo1(0), lo2(0), lo3(0), lo4(0), lo5(0), lo6(0), lo7(0)
        {
        }

        DYN_FUNC uint512_t(uint64_t _lo0, uint64_t _lo1, uint64_t _lo2, uint64_t _lo3)
            : lo0(_lo0), lo1(_lo1), lo2(_lo2), lo3(_lo3),
              lo4(0), lo5(0), lo6(0), lo7(0)
        {
        }

        DYN_FUNC uint512_t(uint64_t _lo0, uint64_t _lo1, uint64_t _lo2, uint64_t _lo3,
                           uint64_t _lo4, uint64_t _lo5, uint64_t _lo6, uint64_t _lo7)
            : lo0(_lo0), lo1(_lo1), lo2(_lo2), lo3(_lo3),
              lo4(_lo4), lo5(_lo5), lo6(_lo6), lo7(_lo7)
        {
        }


        DYN_FUNC bool try_add(const uint512_t& b, uint512_t& res) const
        {
            uint64_t r0, r1, r2, r3, r4, r5, r6, r7;
            uint64_t t, c;

            r0 = lo0 + b.lo0;
            c = (r0 < lo0);
            r1 = lo1 + b.lo1 + c;
            c = (r1 < lo1 || (c && r1 == lo1));
            r2 = lo2 + b.lo2 + c;
            c = (r2 < lo2 || (c && r2 == lo2));
            r3 = lo3 + b.lo3 + c;
            c = (r3 < lo3 || (c && r3 == lo3));
            r4 = lo4 + b.lo4 + c;
            c = (r4 < lo4 || (c && r4 == lo4));
            r5 = lo5 + b.lo5 + c;
            c = (r5 < lo5 || (c && r5 == lo5));
            r6 = lo6 + b.lo6 + c;
            c = (r6 < lo6 || (c && r6 == lo6));
            if (addition_will_overflow(lo7, b.lo7, c))
                return false;
            r7 = lo7 + b.lo7 + c;

            res = {r0, r1, r2, r3, r4, r5, r6, r7};
            return true;
        }

        DYN_FUNC uint512_t operator+(const uint512_t& b) const
        {
            uint64_t r0, r1, r2, r3, r4, r5, r6, r7;
            uint64_t t, c;

            r0 = lo0 + b.lo0;
            c = (r0 < lo0);
            r1 = lo1 + b.lo1 + c;
            c = (r1 < lo1 || (c && r1 == lo1));
            r2 = lo2 + b.lo2 + c;
            c = (r2 < lo2 || (c && r2 == lo2));
            r3 = lo3 + b.lo3 + c;
            c = (r3 < lo3 || (c && r3 == lo3));
            r4 = lo4 + b.lo4 + c;
            c = (r4 < lo4 || (c && r4 == lo4));
            r5 = lo5 + b.lo5 + c;
            c = (r5 < lo5 || (c && r5 == lo5));
            r6 = lo6 + b.lo6 + c;
            c = (r6 < lo6 || (c && r6 == lo6));
            r7 = lo7 + b.lo7 + c;

            return {r0, r1, r2, r3, r4, r5, r6, r7};
        }

        DYN_FUNC bool try_sub(const uint512_t& b, uint512_t& res) const
        {
            if (*this < b)
                return false;
            res = (*this) - b;
            return true;
        }

        DYN_FUNC uint512_t operator-(const uint512_t& b) const
        {
            uint64_t r0, r1, r2, r3, r4, r5, r6, r7;
            uint64_t brw;

            r0 = lo0 - b.lo0;
            brw = (lo0 < b.lo0);
            r1 = lo1 - b.lo1 - brw;
            brw = ((lo1 < b.lo1) || (brw && lo1 == b.lo1)) ? 1 : 0;
            r2 = lo2 - b.lo2 - brw;
            brw = ((lo2 < b.lo2) || (brw && lo2 == b.lo2)) ? 1 : 0;
            r3 = lo3 - b.lo3 - brw;
            brw = ((lo3 < b.lo3) || (brw && lo3 == b.lo3)) ? 1 : 0;
            r4 = lo4 - b.lo4 - brw;
            brw = ((lo4 < b.lo4) || (brw && lo4 == b.lo4)) ? 1 : 0;
            r5 = lo5 - b.lo5 - brw;
            brw = ((lo5 < b.lo5) || (brw && lo5 == b.lo5)) ? 1 : 0;
            r6 = lo6 - b.lo6 - brw;
            brw = ((lo6 < b.lo6) || (brw && lo6 == b.lo6)) ? 1 : 0;
            r7 = lo7 - b.lo7 - brw;

            return {r0, r1, r2, r3, r4, r5, r6, r7};
        }

        DYN_FUNC bool try_mul(const uint512_t& b, uint512_t& res) const
        {
            uint64_t r[8] = {0};

            auto getA = [&](int i) -> uint64_t { return ((&lo0)[i]); };
            auto getB = [&](int i) -> uint64_t { return ((&b.lo0)[i]); };

            auto add128_to = [&](int idx, uint64_t lo, uint64_t hi) -> bool
            {
                uint64_t old = r[idx];
                r[idx] = old + lo;
                uint64_t carry = (r[idx] < old) ? 1 : 0;

                uint64_t to_add = hi + carry;
                int k = idx + 1;
                while (to_add != 0)
                {
                    if (k >= 8)
                        return false;
                    uint64_t oldk = r[k];
                    r[k] = oldk + to_add;
                    if (r[k] < oldk)
                    {
                        to_add = 1;
                    }
                    else
                    {
                        to_add = 0;
                    }
                    k++;
                }
                return true;
            };

            for (int i = 0; i < 8; ++i)
            {
                uint64_t a = getA(i);
                for (int j = 0; j < 8; ++j)
                {
                    uint64_t b_limb = getB(j);
                    uint64_t plo, phi;
                    dyno::mul64x64_to_128(a, b_limb, plo, phi);
                    int idx = i + j;
                    if (idx >= 8 && (plo != 0 || phi != 0))
                        return false;
                    if (idx < 8)
                        if (!add128_to(idx, plo, phi)) return false;
                }
            }
            return true;
        }

        DYN_FUNC bool operator==(const uint512_t& b) const
        {
            return lo0 == b.lo0 && lo1 == b.lo1 && lo2 == b.lo2 && lo3 == b.lo3 &&
                lo4 == b.lo4 && lo5 == b.lo5 && lo6 == b.lo6 && lo7 == b.lo7;
        }

        DYN_FUNC bool operator!=(const uint512_t& b) const { return !(*this == b); }

        DYN_FUNC bool operator<(const uint512_t& b) const
        {
            if (lo7 != b.lo7) return lo7 < b.lo7;
            if (lo6 != b.lo6) return lo6 < b.lo6;
            if (lo5 != b.lo5) return lo5 < b.lo5;
            if (lo4 != b.lo4) return lo4 < b.lo4;
            if (lo3 != b.lo3) return lo3 < b.lo3;
            if (lo2 != b.lo2) return lo2 < b.lo2;
            if (lo1 != b.lo1) return lo1 < b.lo1;
            return lo0 < b.lo0;
        }

        DYN_FUNC bool operator>(const uint512_t& b) const { return b < *this; }
        DYN_FUNC bool operator<=(const uint512_t& b) const { return !(*this > b); }
        DYN_FUNC bool operator>=(const uint512_t& b) const { return !(*this < b); }

        DYN_FUNC uint512_t operator*(const uint512_t& b) const
        {
            uint64_t r[8] = {0};

            auto getA = [&](int i) -> uint64_t { return ((&lo0)[i]); };
            auto getB = [&](int i) -> uint64_t { return ((&b.lo0)[i]); };

            auto add128_to = [&](int idx, uint64_t lo, uint64_t hi)
            {
                uint64_t old = r[idx];
                r[idx] = old + lo;
                uint64_t carry = (r[idx] < old) ? 1 : 0;

                uint64_t to_add = hi + carry;
                int k = idx + 1;
                while (to_add != 0 && k < 8)
                {
                    uint64_t oldk = r[k];
                    r[k] = oldk + to_add;
                    if (r[k] < oldk)
                    {
                        to_add = 1;
                    }
                    else
                    {
                        to_add = 0;
                    }
                    k++;
                }
            };

            for (int i = 0; i < 8; ++i)
            {
                uint64_t a = getA(i);
                for (int j = 0; j < 8; ++j)
                {
                    uint64_t b_limb = getB(j);
                    uint64_t plo, phi;
                    dyno::mul64x64_to_128(a, b_limb, plo, phi);
                    int idx = i + j;
                    if (idx < 8) add128_to(idx, plo, phi);
                }
            }

            return uint512_t(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
        }

        DYN_FUNC uint512_t operator/(const uint512_t& other) const
        {
            return uint512_t(0, 0, 0, 0, 0, 0, 0, 0);
        }
    };

    struct ext_sgn_int512_t
    {
        uint512_t abs_val;
        uint64_t neg_mask;

        DYN_FUNC ext_sgn_int512_t() : abs_val(), neg_mask(0)
        {
        }

        DYN_FUNC ext_sgn_int512_t(const uint512_t& val, bool negative)
            : abs_val(val), neg_mask(negative ? ~0ULL : 0)
        {
        }

        DYN_FUNC ext_sgn_int512_t(const ext_sgn_int256_t& a)
        {
            neg_mask = a.neg_mask;
            abs_val = uint512_t(a.lo0, a.lo1, a.hi0, a.hi1);
        }

        DYN_FUNC bool operator==(const ext_sgn_int512_t& b) const
        {

            bool zero_this = (abs_val == uint512_t());
            bool zero_other = (b.abs_val == uint512_t());
            if (zero_this && zero_other) return true;
            return neg_mask == b.neg_mask && abs_val == b.abs_val;
        }

        DYN_FUNC bool operator!=(const ext_sgn_int512_t& b) const { return !(*this == b); }

        DYN_FUNC bool operator<(const ext_sgn_int512_t& b) const
        {
            bool zero_this = (abs_val == uint512_t());
            bool zero_other = (b.abs_val == uint512_t());
            uint64_t eff_neg = zero_this ? 0 : neg_mask;
            uint64_t eff_b_neg = zero_other ? 0 : b.neg_mask;

            if (eff_neg != eff_b_neg) return eff_neg != 0;
            if (eff_neg == 0) return abs_val < b.abs_val;
            return abs_val > b.abs_val;
        }

        DYN_FUNC bool operator>(const ext_sgn_int512_t& b) const { return b < *this; }
        DYN_FUNC bool operator<=(const ext_sgn_int512_t& b) const { return !(*this > b); }
        DYN_FUNC bool operator>=(const ext_sgn_int512_t& b) const { return !(*this < b); }

        DYN_FUNC ext_sgn_int512_t operator+(const ext_sgn_int512_t& b) const
        {
            if (neg_mask == b.neg_mask)
            {
                return ext_sgn_int512_t(abs_val + b.abs_val, neg_mask != 0);
            }
            else
            {
                if (abs_val >= b.abs_val) return ext_sgn_int512_t(abs_val - b.abs_val, neg_mask != 0);
                else return ext_sgn_int512_t(b.abs_val - abs_val, b.neg_mask != 0);
            }
        }

        DYN_FUNC ext_sgn_int512_t operator-(const ext_sgn_int512_t& b) const
        {
            ext_sgn_int512_t nb = b;
            nb.neg_mask = ~nb.neg_mask;
            return *this + nb;
        }

        DYN_FUNC ext_sgn_int512_t operator*(const ext_sgn_int512_t& b) const
        {
            return ext_sgn_int512_t(abs_val * b.abs_val, neg_mask ^ b.neg_mask);
        }

        DYN_FUNC bool try_add(const ext_sgn_int512_t& other, ext_sgn_int512_t& result) const
        {
            if (neg_mask == other.neg_mask)
            {
                uint512_t abs_result;
                if (!abs_val.try_add(other.abs_val, abs_result))
                    return false;
                result = ext_sgn_int512_t(abs_result, neg_mask != 0);
                return true;
            }
            else
            {
                if (abs_val >= other.abs_val)
                {
                    uint512_t abs_result;
                    if (!abs_val.try_sub(other.abs_val, abs_result))
                        return false;
                    result = ext_sgn_int512_t(abs_result, neg_mask != 0);
                    return true;
                }
                else
                {
                    uint512_t abs_result;
                    if (!other.abs_val.try_sub(abs_val, abs_result))
                        return false;
                    result = ext_sgn_int512_t(abs_result, other.neg_mask != 0);
                    return true;
                }
            }
        }

        DYN_FUNC bool try_sub(const ext_sgn_int512_t& other, ext_sgn_int512_t& result) const
        {
            ext_sgn_int512_t neg_other = other;
            neg_other.neg_mask = ~neg_other.neg_mask;
            return try_add(neg_other, result);
        }

        DYN_FUNC bool try_mul(const ext_sgn_int512_t& other, ext_sgn_int512_t& result) const
        {
            uint512_t abs_result;
            if (!abs_val.try_mul(other.abs_val, abs_result))
                return false;
            result = ext_sgn_int512_t(abs_result, (neg_mask ^ other.neg_mask) != 0);
            return true;
        }

        DYN_FUNC ext_sgn_int512_t operator/(const ext_sgn_int512_t& other) const
        {
            // Handle division by zero
            uint512_t zero = {0, 0, 0, 0, 0, 0, 0, 0};
            if (other.abs_val == zero)
            {
                // Return max value as error
                uint512_t max_val = {~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL};
                return ext_sgn_int512_t(max_val, true);
            }

            uint512_t abs_result = abs_val / other.abs_val;

            bool result_negative = (neg_mask ^ other.neg_mask) != 0;

            return ext_sgn_int512_t(abs_result, result_negative);
        }
    };

#else

    struct uint192_t
    {
        ulonglong3 limbs; // x,y,z correspond to low to high 64-bit parts

        __host__ __device__ uint192_t (

        )
        :
        limbs (make_ulonglong3(0, 0, 0))
        {
        }

        __host__ __device__ uint192_t (uint64_t l0
        ,
        uint64_t l1, uint64_t l2
        )
        :
        limbs (make_ulonglong3(l0, l1, l2))
        {
        }

        __device__ uint192_t operator+(const uint192_t& other) const
        {
            uint192_t result;
            add_uint192_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ uint192_t operator-(const uint192_t& other) const
        {
            uint192_t result;
            sub_uint192_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ uint192_t operator*(const uint192_t& other) const
        {
            uint192_t result;
            mul_uint192_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ bool operator>=(const uint192_t& other) const
        {
            if (limbs.z != other.limbs.z) return limbs.z > other.limbs.z;
            if (limbs.y != other.limbs.y) return limbs.y > other.limbs.y;
            return limbs.x >= other.limbs.x;
        }

        __device__ bool operator<(const uint192_t& other) const
        {
            return !(*this >= other);
        }

        __device__ bool operator==(const uint192_t& other) const = default;
    };

    struct uint256_t
    {
        ulonglong4 limbs; // x,y,z,w correspond to low to high 64-bit parts

        __host__ __device__ uint256_t (

        )
        :
        limbs (make_ulonglong4(0, 0, 0, 0))
        {
        }

        __host__ __device__ uint256_t (uint64_t l0
        ,
        uint64_t l1, uint64_t l2, uint64_t l3
        )
        :
        limbs (make_ulonglong4(l0, l1, l2, l3))
        {
        }

        __device__ uint256_t operator+(const uint256_t& other) const
        {
            uint256_t result;
            add_uint256_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ uint256_t operator-(const uint256_t& other) const
        {
            uint256_t result;
            sub_uint256_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ uint256_t operator*(const uint256_t& other) const
        {
            uint256_t result;
            mul_uint256_vectorized_asm(limbs, other.limbs, result.limbs);
            return result;
        }

        __device__ bool operator>=(const uint256_t& other) const
        {
            if (limbs.w != other.limbs.w) return limbs.w > other.limbs.w;
            if (limbs.z != other.limbs.z) return limbs.z > other.limbs.z;
            if (limbs.y != other.limbs.y) return limbs.y > other.limbs.y;
            return limbs.x >= other.limbs.x;
        }

        __device__ bool operator<(const uint256_t& other) const
        {
            return !(*this >= other);
        }

        __device__ bool operator==(const uint256_t& other) const = default;
    };

#endif
}
#endif
