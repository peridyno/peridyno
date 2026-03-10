
#ifndef BIGINT_BIGINTUTILS_H
#define BIGINT_BIGINTUTILS_H

// Include CUDA headers only when compiling with CUDA
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <cstdint>
#include <cassert>

// Include MSVC specific headers for intrinsics
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "Platform.h"
namespace dyno
{
    // Cross-platform count leading zeros for 64-bit integers
    DYN_FUNC inline int count_leading_zeros(uint64_t x)
    {
#ifdef __CUDA_ARCH__
        return __clzll(x);
#elif defined(_MSC_VER)
        // MSVC implementation
        if (x == 0) return 64;
        unsigned long index;
        _BitScanReverse64(&index, x);
        return 63 - static_cast<int>(index);
#else
        // GCC/Clang and other compilers
        if (x == 0) return 64;
        return __builtin_clzll(x);
#endif
    }

    DYN_FUNC inline int cmp_3_limbs(uint64_t a2, uint64_t a1, uint64_t a0,
                                     uint64_t b2, uint64_t b1, uint64_t b0)
    {
        if (a2 != b2) return (a2 > b2) ? 1 : -1;
        if (a1 != b1) return (a1 > b1) ? 1 : -1;
        if (a0 != b0) return (a0 > b0) ? 1 : -1;
        return 0;
    }
    DYN_FUNC inline void mul64x64_to_128(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi)
    {
#ifndef __CUDA_ARCH__
        uint64_t a0 = (uint32_t)a;
        uint64_t a1 = a >> 32;
        uint64_t b0 = (uint32_t)b;
        uint64_t b1 = b >> 32;

        uint64_t p0 = a0 * b0;
        uint64_t p1 = a0 * b1;
        uint64_t p2 = a1 * b0;
        uint64_t p3 = a1 * b1;

        uint64_t middle = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
        hi = p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32);
        lo = (middle << 32) | (p0 & 0xFFFFFFFFULL);
#else
        asm volatile (
                "{\n\t"
                "mul.lo.u64 %0, %2, %3;\n\t"
                "mul.hi.u64 %1, %2, %3;\n\t"
                "}\n\t"
                : "=l"(lo), "=l"(hi)
                : "l"(a), "l"(b)
            );
#endif
    }
    DYN_FUNC inline void umul64(uint64_t a, uint64_t b,
                       uint64_t& hi, uint64_t& lo)
    {
       mul64x64_to_128(a, b, lo, hi);
    }

    DYN_FUNC inline uint64_t umullo64(uint64_t a, uint64_t b)
    {
        return a * b;
    }

    DYN_FUNC inline uint64_t umulhi64(uint64_t a, uint64_t b)
    {
        uint64_t hi;
#ifndef __CUDA_ARCH__
        uint64_t lo;
        umul64(a, b, hi, lo);
#else
        asm volatile (
                "{\n\t"
                "mul.hi.u64 %0, %1, %2;\n\t"
                "}\n\t"
                : "=l"(hi)
                : "l"(a), "l"(b)
            );
#endif
        return hi;
    }

    DYN_FUNC inline void add128(uint64_t ahi, uint64_t alo,
                       uint64_t bhi, uint64_t blo,
                       uint64_t& rhi, uint64_t& rlo)
    {
#ifndef __CUDA_ARCH__
        rlo = alo + blo;
        rhi = ahi + bhi + (rlo < alo);
#else
        asm volatile (
                "{\n\t"
                "add.cc.u64 %0, %2, %3;\n\t"
                "addc.u64 %1, %4, 0;\n\t"
                "}\n\t"
                : "=l"(rlo), "=l"(rhi)
                : "l"(alo), "l"(blo), "l"(ahi + bhi)
            );
#endif
    }

    DYN_FUNC inline void sub128(uint64_t ahi, uint64_t alo,
                       uint64_t bhi, uint64_t blo,
                       uint64_t& rhi, uint64_t& rlo)
    {
#ifndef __CUDA_ARCH__
        rlo = alo - blo;
        rhi = ahi - bhi - (alo < blo);
#else
        asm volatile (
                "{\n\t"
                "sub.cc.u64 %0, %2, %3;\n\t"
                "subc.u64 %1, %4, 0;\n\t"
                "}\n\t"
                : "=l"(rlo), "=l"(rhi)
                : "l"(alo), "l"(blo), "l"(ahi - bhi)
            );
#endif
    }

    DYN_FUNC inline void
    shr128(uint64_t hi, uint64_t lo, unsigned s, uint64_t& rhi, uint64_t& rlo)
    {
        if (s == 0)
        {
            rhi = hi;
            rlo = lo;
            return;
        }
        if (s >= 128)
        {
            rhi = 0;
            rlo = 0;
            return;
        }
        if (s >= 64)
        {
            unsigned sh = s - 64;
            rhi = 0;
            rlo = hi >> sh;
            return;
        }
        rhi = hi >> s;
        rlo = (hi << (64 - s)) | (lo >> s);
    }

    // ==========================================================
    //  Algorithm 2 : RECIPROCAL_WORD_64
    // ==========================================================
    DYN_FUNC inline uint64_t RECIPROCAL_WORD_64(uint64_t d)
    {
        assert((d >> 63) == 1); // normalized

        uint64_t d0 = d & 1ULL;
        uint64_t d9 = d >> 55; // 2^8 <= d9 < 2^9
        uint64_t d40 = (d >> 24) + 1; // 2^39 <= d40 <= 2^40
        uint64_t d63 = d - (d >> 1);

        const uint64_t NUM_v0 = ((1ULL << 19) - 3ULL * (1ULL << 8));
        uint64_t v0 = NUM_v0 / d9; // 2^10 <= v0 < 2^11
        uint64_t t = v0 * v0 * d40; // so t can be stored in 64 bits
        uint64_t v1 = (v0 << 11) - (t >> 40) - 1; // v1 < 2^22
        t = (1ULL << 60) - v1 * d40;
        uint64_t v2 = (v1 << 13) + ((v1 * t) >> 47);
        uint64_t p_hi, p_lo;
        umul64(v2, d63, p_hi, p_lo);
        uint64_t A_hi = (1ULL << 32), A_lo = 0;
        uint64_t e_hi, e_lo;
        sub128(A_hi, A_lo, p_hi, p_lo, e_hi, e_lo);
        if (d0)
        {
            add128(e_hi, e_lo, 0, (v2 >> 1), e_hi, e_lo);
        }
        assert(e_hi == 0);
        uint64_t e = e_lo;
        uint64_t ve_hi, ve_lo;
        umul64(v2, e, ve_hi, ve_lo);
        uint64_t term = ve_hi >> 1;
        uint64_t v3 = (v2 << 31) + term;
        if (v3 == ~0ULL)
        {
            return v3 - (d << 1);
        }
        uint64_t hi_v3d = umulhi64(v3 + 1, d);
        uint64_t v4 = v3 - (hi_v3d + d);
        return v4;
    }

    DYN_FUNC inline void div128_by_64(uint64_t u1, uint64_t u0,
                             uint64_t d, uint64_t v,
                             uint64_t& q, uint64_t& r)
    {
        assert(u1 < d);
        assert(d & (1ULL << 63));
        uint64_t q_hi, q_lo;
        umul64(v, u1, q_hi, q_lo);

        add128(q_hi, q_lo, u1, u0, q_hi, q_lo);

        uint64_t q1 = q_hi + 1;
        r = u0 - q1 * d;

        if (r > q_lo)
        {
            q1--;
            r += d;
        }
        if (r >= d)
        {
            q1++;
            r -= d;
        }

        q = q1;
    }

    DYN_FUNC inline void div128_by_64_direct(uint64_t u1, uint64_t u0,
                                    uint64_t d,
                                    uint64_t& q, uint64_t& r)
    {
        uint64_t v = RECIPROCAL_WORD_64(d);
        div128_by_64(u1, u0, d, v, q, r);
    }

    DYN_FUNC inline void div128by64_rem(uint64_t hi, uint64_t lo,
                               uint64_t d,
                               uint64_t& q, uint64_t& r)
    {
        assert(d != 0);
        unsigned lz = count_leading_zeros(d);
        unsigned shift = lz;
        uint64_t d_norm = d << shift;
        uint64_t u1_norm = shift == 0 ? hi : ((hi << shift) | (lo >> (64 - shift)));
        uint64_t u0_norm = lo << shift;

        uint64_t v = RECIPROCAL_WORD_64(d_norm);
        div128_by_64(u1_norm, u0_norm, d_norm, v, q, r);
        r >>= shift;
    }

    DYN_FUNC inline void div128by64(uint64_t hi, uint64_t lo,
                           uint64_t d,
                           uint64_t& q)
    {
        assert(d != 0);
        unsigned lz = count_leading_zeros(d);
        unsigned shift = lz;
        uint64_t d_norm = d << shift;
        uint64_t u1_norm = shift ? ((hi << shift) | (lo >> (64 - shift))) : hi;
        uint64_t u0_norm = lo << shift;

        uint64_t v = RECIPROCAL_WORD_64(d_norm);
        uint64_t r;
        div128_by_64(u1_norm, u0_norm, d_norm, v, q, r);
    }

    DYN_FUNC inline void div128by64(uint64_t hi, uint64_t lo, uint64_t d, uint64_t& ret_hi, uint64_t& ret_lo)
    {
        ret_hi = hi / d;
        hi %= d;
        uint64_t r;
        div128by64_rem(hi, lo, d, ret_lo, r);
    }

    DYN_FUNC inline void div128by64_rem(uint64_t hi, uint64_t lo, uint64_t d, uint64_t& ret_hi, uint64_t& ret_lo, uint64_t& r)
    {
        ret_hi = hi / d;
        hi %= d;
        div128by64(hi, lo, d, ret_lo, r);
    }

    DYN_FUNC inline bool greater_than_4_limbs(uint64_t a3, uint64_t a2, uint64_t a1, uint64_t a0, uint64_t b3, uint64_t b2, uint64_t b1, uint64_t b0)
    {
        if (a3 != b3) return a3 > b3;
        if (a2 != b2) return a2 > b2;
        if (a1 != b1) return a1 > b1;
        return a0 >= b0;
    }

    DYN_FUNC inline bool cmp_5_limbs(uint64_t a4, uint64_t a3, uint64_t a2, uint64_t a1, uint64_t a0,
                             uint64_t b4, uint64_t b3, uint64_t b2, uint64_t b1, uint64_t b0)
    {
        if (a4 != b4) return a4 > b4;
        if (a3 != b3) return a3 > b3;
        if (a2 != b2) return a2 > b2;
        if (a1 != b1) return a1 > b1;
        return a0 >= b0;
    }

    DYN_FUNC inline bool addition_will_overflow(uint64_t a, uint64_t b)
    {
        return a > UINT64_MAX - b;
    }
    DYN_FUNC inline bool addition_will_overflow(uint64_t a, uint64_t b, uint64_t c)
    {
        if (a > UINT64_MAX - b) return true;
        uint64_t sum = a + b;
        return sum > UINT64_MAX - c;
    }
    // inline void div192by128_rem(uint64_t u2, uint64_t u1, uint64_t u0, uint64_t d1, uint64_t d0, uint64_t v, uint64_t& q, uint64_t& r1, uint64_t& r0)
    // {
    //     assert(d1 >> 63);
    //     assert(u2 < d1 || (u2 == d1 && u1 < d0));
    //     uint64_t q1, q0;
    //     umul64(v, u2, q1, q0);
    //     add128(q1, q0, u2, u1, q1, q0);
    //     r1 = u1 - q1 * d1;
    //     uint64_t t0, t1;
    //     umul64(d0, q1, t1, t0);
    //
    // }
}

// CUDA-specific vectorized functions - only available when compiling with CUDA
#ifdef __CUDA_ARCH__
#ifdef USE_PTX
__device__ __forceinline__
inline void add_uint128_vectorized_asm(const uint4& a, const uint4& b, uint4& res)
{
    asm volatile ("add.cc.u32      %0, %4, %8;\n\t"
        "addc.cc.u32     %1, %5, %9;\n\t"
        "addc.cc.u32     %2, %6, %10;\n\t"
        "addc.u32        %3, %7, %11;\n\t"
        : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
        "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
}

__device__ __forceinline__
inline void mul_uint128_vectorized_asm(const uint4& a, const uint4& b, uint4& res)
{
    asm volatile ("{\n\t"
        "mul.lo.u32      %0, %4, %8;    \n\t"
        "mul.hi.u32      %1, %4, %8;    \n\t"
        "mad.lo.cc.u32   %1, %4, %9, %1;\n\t"
        "madc.hi.u32     %2, %4, %9,  0;\n\t"
        "mad.lo.cc.u32   %1, %5, %8, %1;\n\t"
        "madc.hi.cc.u32  %2, %5, %8, %2;\n\t"
        "madc.hi.u32     %3, %4,%10,  0;\n\t"
        "mad.lo.cc.u32   %2, %4,%10, %2;\n\t"
        "madc.hi.u32     %3, %5, %9, %3;\n\t"
        "mad.lo.cc.u32   %2, %5, %9, %2;\n\t"
        "madc.hi.u32     %3, %6, %8, %3;\n\t"
        "mad.lo.cc.u32   %2, %6, %8, %2;\n\t"
        "madc.lo.u32     %3, %4,%11, %3;\n\t"
        "mad.lo.u32      %3, %5,%10, %3;\n\t"
        "mad.lo.u32      %3, %6, %9, %3;\n\t"
        "mad.lo.u32      %3, %7, %8, %3;\n\t"
        "}"
        : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
        "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
}

__device__ __forceinline__
inline void add_uint256_vectorized_asm(const ulonglong4& a, const ulonglong4& b, ulonglong4& res)
{
    asm volatile ("add.cc.u64      %0, %4, %8;\n\t"
        "addc.cc.u64     %1, %5, %9;\n\t"
        "addc.cc.u64     %2, %6, %10;\n\t"
        "addc.u64        %3, %7, %11;\n\t"
        : "=l"(res.x), "=l"(res.y), "=l"(res.z), "=l"(res.w)
        : "l"(a.x), "l"(a.y), "l"(a.z), "l"(a.w),
        "l"(b.x), "l"(b.y), "l"(b.z), "l"(b.w));
}

__device__ __forceinline__
inline void sub_uint128_vectorized_asm(const uint4& a, const uint4& b, uint4& res)
{
    asm volatile ("sub.cc.u32      %0, %4, %8;\n\t"
        "subc.cc.u32     %1, %5, %9;\n\t"
        "subc.cc.u32     %2, %6, %10;\n\t"
        "subc.u32        %3, %7, %11;"
        : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
        "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
}

__device__ __forceinline__
inline void sub_uint256_vectorized_asm(const ulonglong4& a, const ulonglong4& b, ulonglong4& res)
{
    asm volatile ("sub.cc.u64      %0, %4, %8;\n\t"
        "subc.cc.u64     %1, %5, %9;\n\t"
        "subc.cc.u64     %2, %6, %10;\n\t"
        "subc.u64        %3, %7, %11;\n\t"
        : "=l"(res.x), "=l"(res.y), "=l"(res.z), "=l"(res.w)
        : "l"(a.x), "l"(a.y), "l"(a.z), "l"(a.w),
        "l"(b.x), "l"(b.y), "l"(b.z), "l"(b.w));
}

__device__ __forceinline__
inline void add_uint192_vectorized_asm(const ulonglong3& a, const ulonglong3& b, ulonglong3& res)
{
    asm volatile ("add.cc.u64      %0, %3, %6;\n\t"
        "addc.cc.u64     %1, %4, %7;\n\t"
        "addc.u64        %2, %5, %8;\n\t"
        : "=l"(res.x), "=l"(res.y), "=l"(res.z)
        : "l"(a.x), "l"(a.y), "l"(a.z),
        "l"(b.x), "l"(b.y), "l"(b.z));
}

__device__ __forceinline__
inline void sub_uint192_vectorized_asm(const ulonglong3& a, const ulonglong3& b, ulonglong3& res)
{
    asm volatile ("sub.cc.u64      %0, %3, %6;\n\t"
        "subc.cc.u64     %1, %4, %7;\n\t"
        "subc.u64        %2, %5, %8;"
        : "=l"(res.x), "=l"(res.y), "=l"(res.z)
        : "l"(a.x), "l"(a.y), "l"(a.z),
        "l"(b.x), "l"(b.y), "l"(b.z));
}

__device__ __forceinline__
inline void mul_uint256_vectorized_asm(const ulonglong4& a, const ulonglong4& b, ulonglong4& res)
{
    asm volatile ("{\n\t"
        "mul.lo.u64      %0, %4, %8;    \n\t"
        "mul.hi.u64      %1, %4, %8;    \n\t"
        "mad.lo.cc.u64   %1, %4, %9, %1;\n\t"
        "madc.hi.u64     %2, %4, %9,  0;\n\t"
        "mad.lo.cc.u64   %1, %5, %8, %1;\n\t"
        "madc.hi.cc.u64  %2, %5, %8, %2;\n\t"
        "madc.lo.u64     %3, %4,%10,  0;\n\t"
        "mad.lo.cc.u64   %2, %4,%10, %2;\n\t"
        "madc.hi.u64     %3, %5, %9, %3;\n\t"
        "mad.lo.cc.u64   %2, %5, %9, %2;\n\t"
        "madc.hi.u64     %3, %6, %8, %3;\n\t"
        "mad.lo.cc.u64   %2, %6, %8, %2;\n\t"
        "madc.lo.u64     %3, %4,%11, %3;\n\t"
        "mad.lo.u64      %3, %5,%10, %3;\n\t"
        "mad.lo.u64      %3, %6, %9, %3;\n\t"
        "mad.lo.u64      %3, %7, %8, %3;\n\t"
        "}"
        : "=l"(res.x), "=l"(res.y), "=l"(res.z), "=l"(res.w)
        : "l"(a.x), "l"(a.y), "l"(a.z), "l"(a.w),
        "l"(b.x), "l"(b.y), "l"(b.z), "l"(b.w));
}

__device__ __forceinline__
inline void mul_uint192_vectorized_asm(const ulonglong3& a, const ulonglong3& b, ulonglong3& res)
{
    asm volatile ("{\n\t"
        "mul.lo.u64      %0, %3, %6;    \n\t"
        "mul.hi.u64      %1, %3, %6;    \n\t"
        "mad.lo.cc.u64   %1, %3, %7, %1;\n\t"
        "madc.hi.u64     %2, %3, %7,  0;\n\t"
        "mad.lo.cc.u64   %1, %4, %6, %1;\n\t"
        "madc.hi.cc.u64  %2, %4, %6, %2;\n\t"
        "madc.lo.u64     %2, %3, %8, %2;\n\t"
        "}"
        : "=l"(res.x), "=l"(res.y), "=l"(res.z)
        : "l"(a.x), "l"(a.y), "l"(a.z),
        "l"(b.x), "l"(b.y), "l"(b.z));
}

#endif
#endif

#endif //BIGINT_BIGINTUTILS_H
