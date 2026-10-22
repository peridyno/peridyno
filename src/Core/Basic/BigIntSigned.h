#ifndef BIGINT_BIGINTSIGNED_H
#define BIGINT_BIGINTSIGNED_H

// Compact signed big-integer types (two's-complement encoding).
//
// Unlike the ext_sgn_* family (defined in BigInt.h / BigInt128.h) these types
// do NOT carry a separate neg_mask word: the value is stored directly in
// two's-complement and the most-significant bit of the highest limb acts as
// the sign bit. This saves one 64-bit word per value.
//
//   * Addition / subtraction are identical to the unsigned versions
//     (modular wrap-around).
//   * Multiplication keeps the low N bits.
//   * Division falls back to unsigned magnitude division with a sign fix-up.
//   * try_add / try_sub / try_mul perform overflow detection.

#include "BigInt.h" // brings in uint128_t / uint192_t / uint256_t and the utils

#ifndef DYN_FUNC
#define DYN_FUNC DYNO_HOST_DEVICE
#endif

namespace dyno
{
#if !DYNO_ENABLE_PTX

    struct int128_t
    {
        uint64_t lo;
        uint64_t hi; // most-significant bit of hi is the sign bit

        DYN_FUNC int128_t() : lo(0), hi(0) {}
        DYN_FUNC int128_t(uint64_t low, uint64_t high) : lo(low), hi(high) {}
        DYN_FUNC int128_t(int64_t a)
            : lo(static_cast<uint64_t>(a)), hi(a < 0 ? ~0ULL : 0ULL) {}

        DYN_FUNC bool is_negative() const { return (hi >> 63) != 0; }
        DYN_FUNC bool is_zero() const { return (lo | hi) == 0; }
        DYN_FUNC int sgn() const { return is_zero() ? 0 : (is_negative() ? -1 : 1); }

        DYN_FUNC int128_t operator-() const
        {
            uint64_t nlo = ~lo + 1ULL;
            uint64_t carry = (nlo == 0) ? 1ULL : 0ULL;
            uint64_t nhi = ~hi + carry;
            return int128_t(nlo, nhi);
        }

        DYN_FUNC uint128_t abs_u() const
        {
            if (!is_negative()) return uint128_t{lo, hi};
            int128_t n = -(*this);
            return uint128_t{n.lo, n.hi};
        }

        DYN_FUNC int128_t operator+(const int128_t& b) const
        {
            uint64_t rlo = lo + b.lo;
            uint64_t carry = (rlo < lo) ? 1ULL : 0ULL;
            uint64_t rhi = hi + b.hi + carry;
            return int128_t(rlo, rhi);
        }

        DYN_FUNC int128_t operator-(const int128_t& b) const
        {
            uint64_t rlo = lo - b.lo;
            uint64_t borrow = (lo < b.lo) ? 1ULL : 0ULL;
            uint64_t rhi = hi - b.hi - borrow;
            return int128_t(rlo, rhi);
        }

        DYN_FUNC int128_t operator*(const int128_t& b) const
        {
            uint128_t prod = uint128_t{lo, hi} * uint128_t{b.lo, b.hi};
            return int128_t(prod.lo, prod.hi);
        }

        DYN_FUNC int128_t operator/(const int128_t& b) const
        {
            uint128_t q = abs_u() / b.abs_u();
            int128_t r(q.lo, q.hi);
            return (is_negative() ^ b.is_negative()) ? -r : r;
        }

        DYN_FUNC bool operator==(const int128_t& b) const { return lo == b.lo && hi == b.hi; }
        DYN_FUNC bool operator!=(const int128_t& b) const { return !(*this == b); }
        DYN_FUNC bool operator<(const int128_t& b) const
        {
            bool na = is_negative(), nb = b.is_negative();
            if (na != nb) return na; // negative < non-negative
            if (hi != b.hi) return hi < b.hi;
            return lo < b.lo;
        }
        DYN_FUNC bool operator>(const int128_t& b) const { return b < *this; }
        DYN_FUNC bool operator<=(const int128_t& b) const { return !(b < *this); }
        DYN_FUNC bool operator>=(const int128_t& b) const { return !(*this < b); }

        DYN_FUNC bool try_add(const int128_t& b, int128_t& res) const
        {
            res = *this + b;
            bool sa = is_negative(), sb = b.is_negative(), sr = res.is_negative();
            return !(sa == sb && sr != sa); // overflow: equal operand signs, different result sign
        }
        DYN_FUNC bool try_sub(const int128_t& b, int128_t& res) const
        {
            res = *this - b;
            bool sa = is_negative(), sb = b.is_negative(), sr = res.is_negative();
            return !(sa != sb && sr != sa);
        }
        DYN_FUNC bool try_mul(const int128_t& b, int128_t& res) const
        {
            bool neg = is_negative() ^ b.is_negative();
            uint128_t prod;
            if (!abs_u().try_mul(b.abs_u(), prod)) return false;
            uint128_t limit = neg ? uint128_t{0ULL, 0x8000000000000000ULL}             // 2^127
                                  : uint128_t{~0ULL, 0x7FFFFFFFFFFFFFFFULL};           // 2^127 - 1
            if (limit < prod) return false;
            int128_t r(prod.lo, prod.hi);
            res = neg ? -r : r;
            return true;
        }
    };

    struct int192_t
    {
        uint64_t limb0;
        uint64_t limb1;
        uint64_t limb2; // most-significant bit of limb2 is the sign bit

        DYN_FUNC int192_t() : limb0(0), limb1(0), limb2(0) {}
        DYN_FUNC int192_t(uint64_t l0, uint64_t l1, uint64_t l2) : limb0(l0), limb1(l1), limb2(l2) {}
        DYN_FUNC int192_t(int64_t a)
            : limb0(static_cast<uint64_t>(a)), limb1(a < 0 ? ~0ULL : 0ULL), limb2(a < 0 ? ~0ULL : 0ULL) {}

        DYN_FUNC bool is_negative() const { return (limb2 >> 63) != 0; }
        DYN_FUNC bool is_zero() const { return (limb0 | limb1 | limb2) == 0; }
        DYN_FUNC int sgn() const { return is_zero() ? 0 : (is_negative() ? -1 : 1); }

        DYN_FUNC int192_t operator-() const
        {
            uint64_t l0 = ~limb0 + 1ULL;
            uint64_t c = (l0 == 0) ? 1ULL : 0ULL;
            uint64_t l1 = ~limb1 + c;
            c = (c && l1 == 0) ? 1ULL : 0ULL;
            uint64_t l2 = ~limb2 + c;
            return int192_t(l0, l1, l2);
        }

        DYN_FUNC uint192_t abs_u() const
        {
            if (!is_negative()) return uint192_t(limb0, limb1, limb2);
            int192_t n = -(*this);
            return uint192_t(n.limb0, n.limb1, n.limb2);
        }

        DYN_FUNC int192_t operator+(const int192_t& b) const
        {
            uint64_t r0 = limb0 + b.limb0;
            uint64_t c = (r0 < limb0) ? 1ULL : 0ULL;
            uint64_t r1 = limb1 + b.limb1 + c;
            c = (r1 < limb1 || (c && r1 == limb1)) ? 1ULL : 0ULL;
            uint64_t r2 = limb2 + b.limb2 + c;
            return int192_t(r0, r1, r2);
        }

        DYN_FUNC int192_t operator-(const int192_t& b) const
        {
            uint64_t r0 = limb0 - b.limb0;
            uint64_t brw = (limb0 < b.limb0) ? 1ULL : 0ULL;
            uint64_t r1 = limb1 - b.limb1 - brw;
            brw = (limb1 < b.limb1 || (brw && limb1 == b.limb1)) ? 1ULL : 0ULL;
            uint64_t r2 = limb2 - b.limb2 - brw;
            return int192_t(r0, r1, r2);
        }

        DYN_FUNC int192_t operator*(const int192_t& b) const
        {
            uint192_t prod = uint192_t(limb0, limb1, limb2) * uint192_t(b.limb0, b.limb1, b.limb2);
            return int192_t(prod.lo, prod.mi, prod.hi);
        }

        DYN_FUNC int192_t operator/(const int192_t& b) const
        {
            uint192_t q = abs_u() / b.abs_u();
            int192_t r(q.lo, q.mi, q.hi);
            return (is_negative() ^ b.is_negative()) ? -r : r;
        }

        DYN_FUNC bool operator==(const int192_t& b) const
        {
            return limb0 == b.limb0 && limb1 == b.limb1 && limb2 == b.limb2;
        }
        DYN_FUNC bool operator!=(const int192_t& b) const { return !(*this == b); }
        DYN_FUNC bool operator<(const int192_t& b) const
        {
            bool na = is_negative(), nb = b.is_negative();
            if (na != nb) return na;
            if (limb2 != b.limb2) return limb2 < b.limb2;
            if (limb1 != b.limb1) return limb1 < b.limb1;
            return limb0 < b.limb0;
        }
        DYN_FUNC bool operator>(const int192_t& b) const { return b < *this; }
        DYN_FUNC bool operator<=(const int192_t& b) const { return !(b < *this); }
        DYN_FUNC bool operator>=(const int192_t& b) const { return !(*this < b); }

        DYN_FUNC bool try_add(const int192_t& b, int192_t& res) const
        {
            res = *this + b;
            bool sa = is_negative(), sb = b.is_negative(), sr = res.is_negative();
            return !(sa == sb && sr != sa);
        }
        DYN_FUNC bool try_sub(const int192_t& b, int192_t& res) const
        {
            res = *this - b;
            bool sa = is_negative(), sb = b.is_negative(), sr = res.is_negative();
            return !(sa != sb && sr != sa);
        }
        DYN_FUNC bool try_mul(const int192_t& b, int192_t& res) const
        {
            bool neg = is_negative() ^ b.is_negative();
            uint192_t prod;
            if (!abs_u().try_mul(b.abs_u(), prod)) return false;
            uint192_t limit = neg ? uint192_t(0ULL, 0ULL, 0x8000000000000000ULL)        // 2^191
                                  : uint192_t(~0ULL, ~0ULL, 0x7FFFFFFFFFFFFFFFULL);     // 2^191 - 1
            if (limit < prod) return false;
            int192_t r(prod.lo, prod.mi, prod.hi);
            res = neg ? -r : r;
            return true;
        }
    };

    struct int256_t
    {
        uint64_t lo0;
        uint64_t lo1;
        uint64_t hi0;
        uint64_t hi1; // most-significant bit of hi1 is the sign bit

        DYN_FUNC int256_t() : lo0(0), lo1(0), hi0(0), hi1(0) {}
        DYN_FUNC int256_t(uint64_t l0, uint64_t l1, uint64_t h0, uint64_t h1)
            : lo0(l0), lo1(l1), hi0(h0), hi1(h1) {}
        DYN_FUNC int256_t(const int128_t& a)
            : lo0(a.lo),
              lo1(a.hi),
              hi0((a.hi >> 63) ? ~0ULL : 0ULL),
              hi1((a.hi >> 63) ? ~0ULL : 0ULL) {}
        DYN_FUNC int256_t(int64_t a)
            : lo0(static_cast<uint64_t>(a)),
              lo1(a < 0 ? ~0ULL : 0ULL), hi0(a < 0 ? ~0ULL : 0ULL), hi1(a < 0 ? ~0ULL : 0ULL) {}

        DYN_FUNC bool is_negative() const { return (hi1 >> 63) != 0; }
        DYN_FUNC bool is_zero() const { return (lo0 | lo1 | hi0 | hi1) == 0; }
        DYN_FUNC int sgn() const { return is_zero() ? 0 : (is_negative() ? -1 : 1); }

        DYN_FUNC int256_t operator-() const
        {
            uint64_t r0 = ~lo0 + 1ULL;
            uint64_t c = (r0 == 0) ? 1ULL : 0ULL;
            uint64_t r1 = ~lo1 + c;
            c = (c && r1 == 0) ? 1ULL : 0ULL;
            uint64_t r2 = ~hi0 + c;
            c = (c && r2 == 0) ? 1ULL : 0ULL;
            uint64_t r3 = ~hi1 + c;
            return int256_t(r0, r1, r2, r3);
        }

        DYN_FUNC uint256_t abs_u() const
        {
            if (!is_negative()) return uint256_t(lo0, lo1, hi0, hi1);
            int256_t n = -(*this);
            return uint256_t(n.lo0, n.lo1, n.hi0, n.hi1);
        }

        DYN_FUNC int256_t operator+(const int256_t& b) const
        {
            uint64_t r0 = lo0 + b.lo0;
            uint64_t c = (r0 < lo0) ? 1ULL : 0ULL;
            uint64_t r1 = lo1 + b.lo1 + c;
            c = (r1 < lo1 || (c && r1 == lo1)) ? 1ULL : 0ULL;
            uint64_t r2 = hi0 + b.hi0 + c;
            c = (r2 < hi0 || (c && r2 == hi0)) ? 1ULL : 0ULL;
            uint64_t r3 = hi1 + b.hi1 + c;
            return int256_t(r0, r1, r2, r3);
        }

        DYN_FUNC int256_t operator-(const int256_t& b) const
        {
            uint64_t r0 = lo0 - b.lo0;
            uint64_t brw = (lo0 < b.lo0) ? 1ULL : 0ULL;
            uint64_t r1 = lo1 - b.lo1 - brw;
            brw = (lo1 < b.lo1 || (brw && lo1 == b.lo1)) ? 1ULL : 0ULL;
            uint64_t r2 = hi0 - b.hi0 - brw;
            brw = (hi0 < b.hi0 || (brw && hi0 == b.hi0)) ? 1ULL : 0ULL;
            uint64_t r3 = hi1 - b.hi1 - brw;
            return int256_t(r0, r1, r2, r3);
        }

        DYN_FUNC int256_t operator*(const int256_t& b) const
        {
            uint256_t prod = uint256_t(lo0, lo1, hi0, hi1) * uint256_t(b.lo0, b.lo1, b.hi0, b.hi1);
            return int256_t(prod.lo0, prod.lo1, prod.hi0, prod.hi1);
        }

        DYN_FUNC int256_t operator/(const int256_t& b) const
        {
            uint256_t q = abs_u() / b.abs_u();
            int256_t r(q.lo0, q.lo1, q.hi0, q.hi1);
            return (is_negative() ^ b.is_negative()) ? -r : r;
        }

        DYN_FUNC bool operator==(const int256_t& b) const
        {
            return lo0 == b.lo0 && lo1 == b.lo1 && hi0 == b.hi0 && hi1 == b.hi1;
        }
        DYN_FUNC bool operator!=(const int256_t& b) const { return !(*this == b); }
        DYN_FUNC bool operator<(const int256_t& b) const
        {
            bool na = is_negative(), nb = b.is_negative();
            if (na != nb) return na;
            if (hi1 != b.hi1) return hi1 < b.hi1;
            if (hi0 != b.hi0) return hi0 < b.hi0;
            if (lo1 != b.lo1) return lo1 < b.lo1;
            return lo0 < b.lo0;
        }
        DYN_FUNC bool operator>(const int256_t& b) const { return b < *this; }
        DYN_FUNC bool operator<=(const int256_t& b) const { return !(b < *this); }
        DYN_FUNC bool operator>=(const int256_t& b) const { return !(*this < b); }

        DYN_FUNC bool try_add(const int256_t& b, int256_t& res) const
        {
            res = *this + b;
            bool sa = is_negative(), sb = b.is_negative(), sr = res.is_negative();
            return !(sa == sb && sr != sa);
        }
        DYN_FUNC bool try_sub(const int256_t& b, int256_t& res) const
        {
            res = *this - b;
            bool sa = is_negative(), sb = b.is_negative(), sr = res.is_negative();
            return !(sa != sb && sr != sa);
        }
        DYN_FUNC bool try_mul(const int256_t& b, int256_t& res) const
        {
            bool neg = is_negative() ^ b.is_negative();
            uint256_t prod;
            if (!abs_u().try_mul(b.abs_u(), prod)) return false;
            uint256_t limit = neg ? uint256_t(0ULL, 0ULL, 0ULL, 0x8000000000000000ULL)        // 2^255
                                  : uint256_t(~0ULL, ~0ULL, ~0ULL, 0x7FFFFFFFFFFFFFFFULL);    // 2^255 - 1
            // prod > limit  <=>  abs_gt_u(prod, limit)
            if (dyno::abs_gt_u(prod.lo0, prod.lo1, prod.hi0, prod.hi1,
                               limit.lo0, limit.lo1, limit.hi0, limit.hi1))
                return false;
            int256_t r(prod.lo0, prod.lo1, prod.hi0, prod.hi1);
            res = neg ? -r : r;
            return true;
        }
    };

#endif // !DYNO_ENABLE_PTX
} // namespace dyno

#endif // BIGINT_BIGINTSIGNED_H
