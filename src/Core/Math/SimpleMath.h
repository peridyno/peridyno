#pragma once
#ifdef CUDA_BACKEND
    #include <cuda_runtime.h>
#endif

#include "Vector.h"
#include "Matrix.h"

namespace dyno
{
    // Only floating point should call these functions.
    template <typename T>
    inline DYN_FUNC int isless  (T const& a, T const& b, T const EPS = EPSILON) { return (a + EPS < b); }
    template <typename T, typename D, typename std::enable_if<!std::is_same<T, D>::value, int>::type = 0>
    inline DYN_FUNC int isless  (T const& a, D const& b, T const EPS = EPSILON) { return isless(a,  static_cast<T>(b), EPS); }

    template <typename T>
    inline DYN_FUNC int isleq   (T const& a, T const& b, T const EPS = EPSILON) { return (a < b + EPS); }
    template <typename T, typename D, typename std::enable_if<!std::is_same<T, D>::value, int>::type = 0>
    inline DYN_FUNC int isleq  (T const& a, D const& b, T const EPS = EPSILON) { return isleq(a,  static_cast<T>(b), EPS); }

    template <typename T>
    inline DYN_FUNC int isgreat (T const& a, T const& b, T const EPS = EPSILON) { return (a > b + EPS); }
    template <typename T, typename D, typename std::enable_if<!std::is_same<T, D>::value, int>::type = 0>
    inline DYN_FUNC int isgreat  (T const& a, D const& b, T const EPS = EPSILON) { return isgreat(a,  static_cast<T>(b), EPS); }

    template <typename T>
    inline DYN_FUNC int isgeq   (T const& a, T const& b, T const EPS = EPSILON) { return (a + EPS > b); }
    template <typename T, typename D, typename std::enable_if<!std::is_same<T, D>::value, int>::type = 0>
    inline DYN_FUNC int isgeq  (T const& a, D const& b, T const EPS = EPSILON) { return isgeq(a,  static_cast<T>(b), EPS); }

    template <typename T>
    inline DYN_FUNC int iseq    (T const& a, T const& b, T const EPS = EPSILON) { return (b < a + EPS && a < b + EPS); }
    template <typename T, typename D, typename std::enable_if<!std::is_same<T, D>::value, int>::type = 0>
    inline DYN_FUNC int iseq  (T const& a, D const& b, T const EPS = EPSILON) { return iseq(a,  static_cast<T>(b), EPS); }
    
    // <:-1 =:0  >:1
    template <typename T>
    inline DYN_FUNC int sign    (T const& a, T const EPS = EPSILON) { return isgreat(a, static_cast<T>(0.f), EPS) - isless(a, static_cast<T>(0.f), EPS);}  

    template <typename T>
    inline DYN_FUNC T clamp(const T& v, const T& lo, const T& hi)
    {
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    template <typename T>
    inline DYN_FUNC Vector<T,2> clamp(const Vector<T,2>& v, const Vector<T,2>& lo, const Vector<T,2>& hi)
    {
        Vector<T,2> ret;
        ret[0] = (v[0] < lo[0]) ? lo[0] : (hi[0] < v[0]) ? hi[0] : v[0];
        ret[1] = (v[1] < lo[1]) ? lo[1] : (hi[1] < v[1]) ? hi[1] : v[1];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 3> clamp(const Vector<T, 3>& v, const Vector<T, 3>& lo, const Vector<T, 3>& hi)
    {
        Vector<T, 3> ret;
        ret[0] = (v[0] < lo[0]) ? lo[0] : (hi[0] < v[0]) ? hi[0] : v[0];
        ret[1] = (v[1] < lo[1]) ? lo[1] : (hi[1] < v[1]) ? hi[1] : v[1];
        ret[2] = (v[2] < lo[2]) ? lo[2] : (hi[2] < v[2]) ? hi[2] : v[2];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 4> clamp(const Vector<T, 4>& v, const Vector<T, 4>& lo, const Vector<T, 4>& hi)
    {
        Vector<T, 3> ret;
        ret[0] = (v[0] < lo[0]) ? lo[0] : (hi[0] < v[0]) ? hi[0] : v[0];
        ret[1] = (v[1] < lo[1]) ? lo[1] : (hi[1] < v[1]) ? hi[1] : v[1];
        ret[2] = (v[2] < lo[2]) ? lo[2] : (hi[2] < v[2]) ? hi[2] : v[2];
        ret[3] = (v[3] < lo[3]) ? lo[3] : (hi[3] < v[3]) ? hi[3] : v[3];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC T abs(const T& v)
    {
        return v < T(0) ? - v : v;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 2> abs(const Vector<T, 2>& v)
    {
        Vector<T, 2> ret;
        ret[0] = (v[0] < T(0)) ? -v[0] : v[0];
        ret[1] = (v[1] < T(0)) ? -v[1] : v[1];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 3> abs(const Vector<T, 3>& v)
    {
        Vector<T, 3> ret;
        ret[0] = (v[0] < T(0)) ? -v[0] : v[0];
        ret[1] = (v[1] < T(0)) ? -v[1] : v[1];
        ret[2] = (v[2] < T(0)) ? -v[2] : v[2];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 4> abs(const Vector<T, 4>& v)
    {
        Vector<T, 3> ret;
        ret[0] = (v[0] < T(0)) ? -v[0] : v[0];
        ret[1] = (v[1] < T(0)) ? -v[1] : v[1];
        ret[2] = (v[2] < T(0)) ? -v[2] : v[2];
        ret[3] = (v[3] < T(0)) ? -v[3] : v[3];

        return ret;
    }

    /**
     * @brief calculate the greatest common divisor of integer a and b
     */
    template <typename Integer>
    inline DYN_FUNC Integer gcd(Integer a, Integer b) {
		while (b != 0) {
			int c = a % b;
			a = b;
			b = c;
		}
		return a;
	}

    template <typename T>
    inline DYN_FUNC T minimum(const T& v0, const T& v1)
    {
        return v0 < v1 ? v0 : v1;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 2> minimum(const Vector<T, 2>& v0, const Vector<T, 2>& v1)
    {
        Vector<T, 2> ret;
        ret[0] = (v0[0] < v1[0]) ? v0[0] : v1[0];
        ret[1] = (v0[1] < v1[1]) ? v0[1] : v1[1];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 3> minimum(const Vector<T, 3>& v0, const Vector<T, 3>& v1)
    {
        Vector<T, 3> ret;
        ret[0] = (v0[0] < v1[0]) ? v0[0] : v1[0];
        ret[1] = (v0[1] < v1[1]) ? v0[1] : v1[1];
        ret[2] = (v0[2] < v1[2]) ? v0[2] : v1[2];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 4> minimum(const Vector<T, 4>& v0, const Vector<T, 4>& v1)
    {
        Vector<T, 4> ret;
        ret[0] = (v0[0] < v1[0]) ? v0[0] : v1[0];
        ret[1] = (v0[1] < v1[1]) ? v0[1] : v1[1];
        ret[2] = (v0[2] < v1[2]) ? v0[2] : v1[2];
        ret[3] = (v0[3] < v1[3]) ? v0[3] : v1[3];

        return ret;
    }


    template <typename T>
    inline DYN_FUNC T maximum(const T& v0, const T& v1)
    {
        return v0 > v1 ? v0 : v1;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 2> maximum(const Vector<T, 2>& v0, const Vector<T, 2>& v1)
    {
        Vector<T, 2> ret;
        ret[0] = (v0[0] > v1[0]) ? v0[0] : v1[0];
        ret[1] = (v0[1] > v1[1]) ? v0[1] : v1[1];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 3> maximum(const Vector<T, 3>& v0, const Vector<T, 3>& v1)
    {
        Vector<T, 3> ret;
        ret[0] = (v0[0] > v1[0]) ? v0[0] : v1[0];
        ret[1] = (v0[1] > v1[1]) ? v0[1] : v1[1];
        ret[2] = (v0[2] > v1[2]) ? v0[2] : v1[2];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 4> maximum(const Vector<T, 4>& v0, const Vector<T, 4>& v1)
    {
        Vector<T, 4> ret;
        ret[0] = (v0[0] > v1[0]) ? v0[0] : v1[0];
        ret[1] = (v0[1] > v1[1]) ? v0[1] : v1[1];
        ret[2] = (v0[2] > v1[2]) ? v0[2] : v1[2];
        ret[3] = (v0[3] > v1[3]) ? v0[3] : v1[3];

        return ret;
    }

    template <typename T>
    inline DYN_FUNC T dot(Vector<T, 2> const& U, Vector<T, 2> const& V)
    {
        return U[0] * V[0] + U[1] * V[1];
    }

    template <typename T>
    inline DYN_FUNC T dot(Vector<T, 3> const& U, Vector<T, 3> const& V)
    {
        return U[0] * V[0] + U[1] * V[1] + U[2] * V[2];
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 3> cross(Vector<T, 3> const& U, Vector<T, 3> const& V)
    {
        return Vec3f{
            U[1] * V[2] - U[2] * V[1],
            U[2] * V[0] - U[0] * V[2],
            U[0] * V[1] - U[1] * V[0]
        };
    }

    template <typename T>
    inline DYN_FUNC T dotcross(Vector<T, 3> const& U, Vector<T, 3> const& V, Vector<T, 3> const& W)
    {
        // U (V x W)
        return dot(U, cross(V, W));
    }

    template <typename T>
    inline DYN_FUNC Vector<T, 2> perp(Vector<T, 2> const& v)
    {
        return Vector<T, 2> {v[1], -v[0]};
    }

    // Cross product of 2D vectors, signed area of the parallelogram 
    /*
    //       --------
    //      /       /
    //     v1  S   /
    //    /       /
    //    ---v0---
    //      
    */
    template <typename T>
    inline DYN_FUNC T dotperp(Vector<T, 2> const& v0, Vector<T, 2> const& v1)
    {
        return dot(v0, perp(v1));
    }

    // return the lowest bit of x (1<<y)
    inline DYN_FUNC unsigned int lowbit(unsigned int x) {return x & (-x);}

    // return how many 1-bit in x 
    inline DYN_FUNC unsigned int countbit(unsigned int x) { unsigned int cnt = 0; while(x) {x -= lowbit(x); cnt++;} return cnt;}
    
    // Count Leading Zeros
    inline DYN_FUNC unsigned int builtin_clz(unsigned int x) {
        unsigned int r = 0;
        if (!(x & 0xFFFF0000)) r += 16, x <<= 16;
        if (!(x & 0xFF000000)) r += 8,  x <<= 8;
        if (!(x & 0xF0000000)) r += 4,  x <<= 4;
        if (!(x & 0xC0000000)) r += 2,  x <<= 2;
        if (!(x & 0x80000000)) r += 1,  x <<= 1;
        return r;
    }

    // Most Significant Bit: return y for 1 << y ( assert unsigned int is 32-bits)
    inline DYN_FUNC unsigned int MSB(unsigned int x) {return 32 - builtin_clz(x);}    

    // return the id of lowest bit of x
    inline DYN_FUNC unsigned int lownum(unsigned int x) {return MSB(lowbit(x));}

    // check bit (x & 1<<y)
    inline DYN_FUNC int checkbit(unsigned int const&x, unsigned int const& y) {return (x >> y) & 1u;}

}