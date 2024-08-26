#include "Platform.h"

#include "Matrix.h"
#include "Vector/Vector2D.h"
#include "Vector.h"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace dyno
{
    namespace cgeo
    {	
        #define REAL_infinity 1.0e30
        #define REAL_ZERO 1.0e-5
        #define REAL_EPS 1e-4
        #define	REAL_EQUAL(a,b)  (((a < b + REAL_EPS) && (a > b - REAL_EPS)) ? true : false)
        #define REAL_GREAT(a,b) ((a > REAL_EPS + b)? true: false) 
        #define REAL_LESS(a,b) ((a + REAL_EPS < b)? true: false) 

        template<typename T>
        DYN_FUNC void Swap(T& a, T& b) { T c = a; a = b; b = c;}
        DYN_FUNC float Dot(Vec3f const& U, Vec3f const& V)
        {
            float dot = U[0] * V[0] + U[1] * V[1] + U[2] * V[2];
            return dot;
        }

        DYN_FUNC float Dot(Vec2f const& U, Vec2f const& V)
        {
            float dot = U[0] * V[0] + U[1] * V[1];
            return dot;
        }

        DYN_FUNC Vec3f Cross(Vec3f const& U, Vec3f const& V)
        {
            Vec3f cross =
            {
                U[1] * V[2] - U[2] * V[1],
                U[2] * V[0] - U[0] * V[2],
                U[0] * V[1] - U[1] * V[0]
            };
            return cross;
        }

        // U (V x W)
        DYN_FUNC float DotCross(Vec3f const& U, Vec3f const& V,
            Vec3f const& W)
        {
            return Dot(U, Cross(V, W));
        }

        DYN_FUNC Vec2f Perp(Vec2f const& v)
        {
            return Vec2f(v[1], -v[0]);
        }

        // 2D Cross
        // Dot((x0,x1),Perp(y0,y1)) = x0*y1 - x1*y0
        DYN_FUNC float DotPerp(Vec2f const& v0, Vec2f const& v1)
        {
            return Dot(v0, Perp(v1));
        }

        DYN_FUNC bool Sign(Vec3f const& n, Vec3f const& p0, Vec3f const& p1, Vec3f const& p2)
        {
            return REAL_GREAT(Dot(n, Cross(p1 - p0, p2 - p0)), 0.f);
        }

        DYN_FUNC bool isOverLap(float& c0, float& c1, float a0, float a1, float b0, float b1)
        {
            c0 = fmax(a0, b0);
            c1 = fmin(a1, b1);
            if (REAL_LESS(c1, c0)) return false;
            return true;
        }

        // get closest point v from point p to triangle a0a1a2 of minimum distance
        DYN_FUNC Vec3f getProjectionVF(Vec3f p,
                            Vec3f a0, Vec3f a1, Vec3f a2)
        {
            //triangle base: a0, e0(a0,a1), e1(a0,a2)
            //point base: p
            //projection base: v (para0, para1, para2)

            Vec3f d = a0 - p;
            Vec3f e0 = a1 - a0;
            Vec3f e1 = a2 - a0;
            float a00 = e0.dot(e0);
            float a01 = e0.dot(e1);
            float a11 = e1.dot(e1);
            float b0 = e0.dot(d);
            float b1 = e1.dot(d);
            float f = d.dot(d);
            float det = fmax(a00 * a11 - a01 * a01, 0.f);
            float s = a01 * b1 - a11 * b0;
            float t = a01 * b0 - a00 * b1;
            if (s + t <= det) {
                if (REAL_LESS(s, 0.f)) {
                    if (REAL_LESS(t, 0.f)) {
                        //region 4
                        if (REAL_LESS(b0,0.f)){
                            t = 0.f;
                            if (-b0 >= a00){
                                s = 1.f;
                            }
                            else{
                                s = -b0 / a00;
                            }
                        }
                        else{
                            s = 0.f;
                            if (REAL_GREAT(b1,0.f)||REAL_EQUAL(b1,0.f)){
                                t = 0.f;
                            }
                            else if (REAL_GREAT(-b1,a11)|| REAL_EQUAL(-b1,a11)){
                                t = 1.f;
                            }
                            else{
                                t = -b1 / a11;
                            }
                        }
                    }
                    else {
                        //region 3
                        s = 0.f;
                        if (REAL_GREAT(b1,0.f)||REAL_EQUAL(b1,0.f)){
                            t = 0.f;
                        }
                        else if (REAL_GREAT(-b1,a11)||REAL_EQUAL(-b1,a11)){
                            t = 1.f;
                        }
                        else{
                            t = -b1 / a11;
                        }
                    }
                }
                else if (REAL_LESS(t, 0.f)) {
                    //region 5
                    t = 0.f;
                    if (REAL_GREAT(b0,0.f)||REAL_EQUAL(b0,0.f)){
                        s = 0.f;
                    }
                    else if (REAL_GREAT(-b0,a00)||REAL_EQUAL(-b0,a00)){
                        s = 1.f;
                    }
                    else{
                        s = -b0 / a00;
                    }
                }
                else {
                    //region 0, minimum at interior point
                    s /= det;
                    t /= det;
                }
            }
            else {
                float tmp0 = 0.f; float tmp1 = 0.f; float numer = 0.f; float denom = 0.f;
                if (REAL_LESS(s, 0.f)) {
                        //region 2
                    tmp0 = a01 + b0;
                    tmp1 = a11 + b1;
                    if (REAL_GREAT(tmp1,tmp0)){
                        numer = tmp1 - tmp0;
                        denom = a00 - 2.f * a01 + a11;
                        if (REAL_GREAT(numer,denom)||REAL_EQUAL(numer,denom)){
                            s = 1.f;
                            t = 0.f;
                        }
                        else{
                            s = numer / denom;
                            t = 1.f - s;
                        }
                    }
                    else
                    {
                        s = 0.f;
                        if (REAL_LESS(tmp1,0.f)||REAL_EQUAL(tmp1,0.f)){
                            t = 1.f;
                        }
                        else if (REAL_GREAT(b1,0.f)||REAL_EQUAL(b1,0.f)){
                            t = 0.f;
                        }
                        else{
                            t = -b1 / a11;
                        }
                    }
                }
                else if (REAL_LESS(t, 0.f)) {
                        //region 6
                    tmp0 = a01 + b1;
                    tmp1 = a00 + b0;
                    if (REAL_GREAT(tmp1,tmp0)){
                        numer = tmp1 - tmp0;
                        denom = a00 - 2.f * a01 + a11;
                        if (REAL_GREAT(numer,denom)||REAL_EQUAL(numer,denom)){
                            t = 1.f;
                            s = 0.f;
                        }
                        else
                        {
                            t = numer / denom;
                            s = 1.f - t;
                        }
                    }
                    else{
                        t = 1.f;
                        if (REAL_LESS(tmp1,0.f)||REAL_EQUAL(tmp1,0.f)){
                            s = 1.f;
                        }
                        else if (REAL_GREAT(b0,0.f)||REAL_EQUAL(b0,0.f)){
                            s = 0.f;
                        }
                        else{
                            s = -b0 / a00;
                        }
                    }
                }
                else {
                    //region 1
                    numer = a11 + b1 - a01 - b0;
                    if (REAL_LESS(numer,0.f)||REAL_EQUAL(numer,0.f)){
                        s = 0.f;
                        t = 1.f;
                    }
                    else{
                        denom = a00 - 2.f * a01 + a11;
                        if (REAL_GREAT(numer,denom)||REAL_EQUAL(numer,denom)){
                            s = 1.f;
                            t = 0.f;
                        }
                        else{
                            s = numer / denom;
                            t = 1.f - s;
                        }
                    }
                }
            }

            Vec3f v = (a0 + s * e0 + t * e1);
            // if (para != nullptr) {
            //     para[0] = 1.0;
            //     para[1] = 1.0-s-t;
            //     para[2] = s;
            //     para[3] = t;
            // }            
            return v;
        }      

        // get unit direction d from point p to triangle a0a1a2 of minimum distance
        DYN_FUNC Vec3f getDirectionVF(Vec3f p,
                            Vec3f a0, Vec3f a1, Vec3f a2)
        {
            Vec3f d = getProjectionVF(p, a0, a1, a2) - p;
            return d / d.norm();
        }

        // get minimum distance from point p to triangle a0a1a2
        DYN_FUNC float getDistanceVF(Vec3f p,
                            Vec3f a0, Vec3f a1, Vec3f a2)
        {
            // return sqrt(Dot(a0 - p, a0 - p)); // DEBUG

            Vec3f d = getProjectionVF(p, a0, a1, a2) - p;
            return sqrt(Dot(d,d));
        }

        // check tet b0b1b2b3 whether in narrow band of triangle a0a1a2 (Dis < Band Width)
        // But, this function is an approximate method, only check center point of tet.
        DYN_FUNC bool isInNarrowBand(Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3,
                        Vec3f a0, Vec3f a1, Vec3f a2, float d)
        {
            return REAL_LESS(
                    getDistanceVF((b0 + b1 + b2 + b3) / 4.f, a0, a1, a2)
                    , d);
        }

        // Test for intersectino of two Convex coplanar Polygon. (2D Separating Axis)
        inline DYN_FUNC bool isConPolyOverLap2D(Vec3f face_n, int n_a, Vec3f* a,
                        int n_b, Vec3f* b)
        {
            if (n_a == 0 || n_b == 0) return false;
            auto project = [&](Vec3f axis, Vec3f* p, int n, float& pmin, float& pmax)
            {
                pmin = REAL_infinity;
                pmax = -REAL_infinity;
                for (int i = 0; i < n; i++)
                {
                    float dot = Dot(axis, p[i] - Vec3f(0.0));
                    pmin = fmin(pmin, dot);
                    pmax = fmax(pmax, dot);
                }
            };
            auto check = [&](Vec3f axis)
            {
                float amin, amax, bmin, bmax, cmin, cmax;
                project(axis, a, n_a, amin, amax);
                project(axis, b, n_b, bmin, bmax);
                if(!isOverLap(cmin, cmax, amin, amax, bmin, bmax))
                {
                    return false;
                }
                return true;
            };
            int j;
            // Check each edge noraml
            for (int i = 0; i < n_a; i++)
            {
                j = (i + 1 == n_a) ? 0 : i + 1;
                Vec3f axis = Cross(face_n, a[j] - a[i]);
                if(!check(axis)) 
                {
                    // printf("[geo] [%d]Polygon((%f, %f, %f), (%f, %f, %f), (%f, %f, %f))\n [%d] Polygon((%f, %f, %f),(%f, %f, %f),(%f, %f, %f),(%f, %f, %f))\n", 
                    //         n_a, a[0][0], a[0][1], a[0][2],
                    //         a[1][0], a[1][1], a[1][2],
                    //         a[2][0], a[2][1], a[2][2],
                    //         n_b, b[0][0], b[0][1], b[0][2],
                    //         b[1][0], b[1][1], b[1][2],
                    //         b[2][0], b[2][1], b[2][2],
                    //         b[3][0], b[3][1], b[3][2]);
                    return false;
                }
            }

            for (int i = 0; i < n_b; i++)
            {
                j = (i + 1 == n_b) ? 0 : i + 1;
                Vec3f axis = Cross(face_n, b[j] - b[i]);
                if(!check(axis)) 
                {
                    // printf("[geo] [%d]Polygon((%f, %f, %f), (%f, %f, %f), (%f, %f, %f))\n [%d] Polygon((%f, %f, %f),(%f, %f, %f),(%f, %f, %f),(%f, %f, %f))\n", 
                    //         n_a, a[0][0], a[0][1], a[0][2],
                    //         a[1][0], a[1][1], a[1][2],
                    //         a[2][0], a[2][1], a[2][2],
                    //         n_b, b[0][0], b[0][1], b[0][2],
                    //         b[1][0], b[1][1], b[1][2],
                    //         b[2][0], b[2][1], b[2][2],
                    //         b[3][0], b[3][1], b[3][2]);                 
                    return false;
                }
            }
            // printf("[geo] [%d]Polygon((%f, %f, %f), (%f, %f, %f), (%f, %f, %f))\n [%d] Polygon((%f, %f, %f),(%f, %f, %f),(%f, %f, %f),(%f, %f, %f))\n", 
            //                 n_a, a[0][0] * 10, a[0][1] * 10, a[0][2] * 10,
            //                 a[1][0] * 10, a[1][1] * 10, a[1][2] * 10,
            //                 a[2][0] * 10, a[2][1] * 10, a[2][2] * 10,
            //                 n_b, b[0][0] * 10, b[0][1] * 10, b[0][2] * 10,
            //                 b[1][0] * 10, b[1][1] * 10, b[1][2] * 10,
            //                 b[2][0] * 10, b[2][1] * 10, b[2][2] * 10,
            //                 b[3][0] * 10, b[3][1] * 10, b[3][2] * 10);
            return true;
        }

        inline DYN_FUNC float getOverLapBoxAreaInPoly2D(Vec3f face_n, int n_a, Vec3f* a,
                        int n_b, Vec3f* b)
        {
            float res = REAL_infinity;
            if (n_a == 0 || n_b == 0) return 0.f;
            auto project = [&](Vec3f axis, Vec3f* p, int n, float& pmin, float& pmax)
            {
                pmin = REAL_infinity;
                pmax = -REAL_infinity;
                for (int i = 0; i < n; i++)
                {
                    float dot = Dot(axis, p[i] - Vec3f(0.0));
                    pmin = fmin(pmin, dot);
                    pmax = fmax(pmax, dot);
                }
            };

            auto cal_box_area = [&](Vec3f axis)
            {
                float amin, amax, bmin, bmax, cmin, cmax;
                project(axis, a, n_a, amin, amax);
                project(axis, b, n_b, bmin, bmax);
                if(!isOverLap(cmin, cmax, amin, amax, bmin, bmax))
                {
                    return 0.f;
                }

                float dx = cmax - cmin;
                Vec3f axis_y = Cross(face_n, axis);
                axis_y.normalize();
                project(axis_y, a, n_a, amin, amax);
                project(axis_y, b, n_b, bmin, bmax);
                if(!isOverLap(cmin, cmax, amin, amax, bmin, bmax))
                {
                    return 0.f;
                }
                float dy = cmax - cmin;
                // printf("[geo] a %f\n", dx * dy);
                return dx * dy;
            };

            int j;
            // Check each edge noraml
            for (int i = 0; i < n_a; i++)
            {
                j = (i + 1 == n_a) ? 0 : i + 1;
                Vec3f axis = Cross(face_n, a[j] - a[i]);
                axis.normalize();
                float area = cal_box_area(axis);
                res = fmin(res, area);
                if(REAL_EQUAL(area, 0.f)) 
                {
                    // printf("[geo] [%d]Polygon((%f, %f, %f), (%f, %f, %f), (%f, %f, %f))\n [%d] Polygon((%f, %f, %f),(%f, %f, %f),(%f, %f, %f),(%f, %f, %f))\n", 
                    //         n_a, a[0][0], a[0][1], a[0][2],
                    //         a[1][0], a[1][1], a[1][2],
                    //         a[2][0], a[2][1], a[2][2],
                    //         n_b, b[0][0], b[0][1], b[0][2],
                    //         b[1][0], b[1][1], b[1][2],
                    //         b[2][0], b[2][1], b[2][2],
                    //         b[3][0], b[3][1], b[3][2]);
                    return 0.f;
                }
            }

            for (int i = 0; i < n_b; i++)
            {
                j = (i + 1 == n_b) ? 0 : i + 1;
                Vec3f axis = Cross(face_n, b[j] - b[i]);
                axis.normalize();
                float area = cal_box_area(axis);
                res = fmin(res, area);
                if(REAL_EQUAL(area, 0.f)) 
                {
                    // printf("[geo] [%d]Polygon((%f, %f, %f), (%f, %f, %f), (%f, %f, %f))\n [%d] Polygon((%f, %f, %f),(%f, %f, %f),(%f, %f, %f),(%f, %f, %f))\n", 
                    //         n_a, a[0][0], a[0][1], a[0][2],
                    //         a[1][0], a[1][1], a[1][2],
                    //         a[2][0], a[2][1], a[2][2],
                    //         n_b, b[0][0], b[0][1], b[0][2],
                    //         b[1][0], b[1][1], b[1][2],
                    //         b[2][0], b[2][1], b[2][2],
                    //         b[3][0], b[3][1], b[3][2]);                 
                    return 0.f;
                }
            }

            // printf("[geo] Area %f\n", res);
            return res;
        }

        // Test for intersectino of a triangle and a tetrahedron that have a shared vertex.
        inline DYN_FUNC bool isIntrTri2Tet(Vec3f a0, Vec3f a1, Vec3f a2,
                        Vec3f b1, Vec3f b2, Vec3f b3) // b0 = a0
        {
            
            // Intersection with Plane A -> <= 4 point Convex Polygon A
            Vec3f nA = (a1 - a0).cross(a2- a0); // face normal
            nA.normalize();
            Vec3f oA = a0;
            Vec3f t[3] = {b1, b2, b3};
            Vec3f p[3];
            int p_num = 0;

            float d[3]; int s[3];
            for (int i = 0; i < 3; i++) {
                d[i] = Dot(nA, t[i] - oA);
                s[i] = REAL_EQUAL(d[i], 0.f) ? 0 : (REAL_LESS(d[i], 0.f) ? -1 : 1);
            }

            // Tet degrade to tri
            if (s[0] == 0 && s[1] == 0 && s[2] == 0) return false;

            for (int i = 0; i < 3; i++)
            {
                if(s[i] == 0) p[p_num++] = t[i];
                else
                {
                    for (int j = i + 1; j < 3; j++)
                    if (s[i] * s[j] < 0) p[p_num++] = t[j] + (t[i] - t[j]) * d[j] / (d[j] - d[i]);
                }
            }
            if (p_num > 2) printf("[geo] Error: p_num > 2 %d %d %d %f %f %f\n", s[0], s[1], s[2], d[0], d[1], d[2]);
            if (p_num != 2) return false;

            Vec3f vec_a1 = (a1 - oA);
            Vec3f vec_a2 = (a2 - oA);
            Vec3f vec_p1 = (p[0] - oA);
            Vec3f vec_p2 = (p[1] - oA);
            vec_a1.normalize();
            vec_a2.normalize();
            Vec3f mid_a = (vec_a1 + vec_a2) * 0.5f;
            mid_a.normalize();
            Real cos_a = Dot(vec_a1, mid_a);
            Real cos_p1 = Dot(vec_p1, mid_a);
            Real cos_p2 = Dot(vec_p2, mid_a);
            if(REAL_LESS(cos_a, cos_p1) || REAL_LESS(cos_a, cos_p2))
            {
                    // printf("[CSR] No Intr Polygon((%f, %f, %f), (%f, %f, %f), (%f, %f, %f))\nPyramid(Polygon((%f, %f, %f),(%f, %f, %f),(%f, %f, %f)),(%f, %f, %f))\n",
                    //     a0[0] * 10, a0[1] * 10, a0[2] * 10,
                    //     a1[0] * 10, a1[1] * 10, a1[2] * 10,
                    //     a2[0] * 10, a2[1] * 10, a2[2] * 10,
                    //     a0[0] * 10, a0[1] * 10, a0[2] * 10,
                    //     b1[0] * 10, b1[1] * 10, b1[2] * 10,
                    //     b2[0] * 10, b2[1] * 10, b2[2] * 10,
                    //     b3[0] * 10, b3[1] * 10, b3[2] * 10);
                return true;
            }
            return false;
        }

        // Test for intersectino of a triangle and a tetrahedron.
        inline DYN_FUNC bool isIntrTri2Tet(Vec3f a0, Vec3f a1, Vec3f a2,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3)
        {
            // Intersection with Plane A -> <= 4 point Convex Polygon A
            Vec3f nA = (a1 - a0).cross(a2 - a0); // face normal
            nA.normalize();
            Vec3f oA = a0;
            Vec3f t[4] = {b0, b1, b2, b3};
            Vec3f p[4];
            int p_num = 0;

            float d[4]; int s[4];
            for (int i = 0; i < 4; i++) {
                d[i] = Dot(nA, t[i] - oA);
                if(d[i] != d[i]) 
                printf("[geo] Error d nan nA(%f %f %f) oA(%f %f %f) t(%f %f %f)\n", nA[0], nA[1], nA[2], oA[0], oA[1], oA[2], t[i][0], t[i][1], t[i][2]);
                s[i] = REAL_EQUAL(d[i], 0.f) ? 0 : (REAL_LESS(d[i], 0.f) ? -1 : 1);
            }
            for (int i = 0; i < 4; i++)
            {
                if(s[i] == 0) p[p_num++] = t[i];
                else
                {
                    for (int j = i + 1; j < 4; j++)
                    if (s[i] * s[j] < 0) {
                        if (REAL_EQUAL(d[j] - d[i], 0.f))
                            p[p_num++] = t[i];
                        else
                            p[p_num++] = t[j] + (t[i] - t[j]) * d[j] / (d[j] - d[i]);
                    }
                }
            }
            // if(p_num == 2)
            // {
            //     printf("[geo] d0(%f) d1(%f) d2(%f) d3(%f)\n", d[0],d[1],d[2],d[3]);
            // }
            if (p_num > 4) printf("[geo] Error: p_num > 4 [%d %d %d %d]\n", s[0], s[1], s[2], s[3]);
            // set order.
            if (p_num == 4)
            {
                bool s0 = Sign(nA, p[0], p[1], p[2]);
                bool s1 = Sign(nA, p[0], p[1], p[3]);
                if(s1 != s0) Swap(p[0], p[3]);
                else{
                    bool s2 = Sign(nA, p[0], p[2], p[3]);
                    if(s2 != s0) Swap(p[2], p[3]);
                }
                // if(Sign(nA, p[0], p[1], p[2]) != Sign(nA, p[0], p[2], p[3])) printf("[geo] Error in p order\n");
            }

            if (p_num > 0)
            {
                for(int i = 0; i < p_num; i++)
                {
                    // if(!isInTet(p[i], b0, b1, b2, b3))
                    // {
                    //     printf("[geo] Error outside tet\n");
                    // }
                    if(!(REAL_GREAT(Dot(nA, p[i] - oA), -REAL_ZERO) && REAL_LESS(Dot(nA, p[i] - oA), REAL_ZERO) ))                    
                    {
                        printf("[geo] (isIntrTri2Tet) Error outside plane %f (%f %f %f) s[%d %d %d %d] d(%f %f %f %f)\n", Dot(nA, p[i] - oA), nA[0], p[i][0], oA[0], s[0], s[1], s[2], s[3], d[0], d[1], d[2], d[3]);
                    }
                }
            }
            // printf("[geo] p_num: %d\n", p_num);
            // Check if Triangle A is overlap with Polygon A
            Vec3f a[3] = {a0, a1, a2};
            return isConPolyOverLap2D(nA, 3, a, p_num, p);
        }

        // Test for containment of a point by a tetrahedron.
        DYN_FUNC bool isInTet(Vec3f p,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3)
        {
            Vec3f edge20, edge10, edge30, edge21, edge31;
            Vec3f diffP0, diffP1;
            float tsp = 0.f, zero = 0.f;
            float tps0 = 0.f;

            // face <0,2,1>
            edge20 = b2 - b0;
            edge10 = b1 - b0;
            edge30 = b3 - b0;
            tps0 = DotCross(edge30, edge10, edge20); // Two cases: +[0,1,2,3] -[0,2,1,3]
        
            diffP0 = p - b0;
            tsp = DotCross(diffP0, edge10, edge20);
            if (REAL_LESS(tsp * tps0, zero))
            {
                return false;
            }

            // face <0,1,3>
            tsp = DotCross(diffP0, edge30, edge10);
            if (REAL_LESS(tsp * tps0, zero))
            {
                return false;
            }

            // face <0,3,2>
            tsp = DotCross(diffP0, edge20,  edge30);
            if (REAL_LESS(tsp * tps0, zero))
            {
                return false;
            }

            // face<1,2,3>
            edge21 = b2 - b1;
            edge31 = b3 - b1;
            diffP1 = p - b1;
            tsp = DotCross(diffP1, edge31, edge21);
            if (REAL_LESS(tsp * tps0, zero))
            {
                return false;
            }
            // printf("[Geo]Tet Pyramid(Polygon((%f, %f, %f),(%f, %f, %f),(%f, %f, %f)),(%f, %f, %f)) , (%f, %f, %f)\n", 
            //             b0[0] * 10, b0[1] * 10, b0[2] * 10,
            //             b1[0] * 10, b1[1] * 10, b1[2] * 10,
            //             b2[0] * 10, b2[1] * 10, b2[2] * 10,
            //             b3[0] * 10, b3[1] * 10, b3[2] * 10,
            //             p[0] * 10, p[1] * 10, p[2] * 10);
            return true;
        }

        DYN_FUNC float getVolume(Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3)
        {
            Vec3f edge20, edge10, edge30;
            edge20 = b2 - b0;
            edge10 = b1 - b0;
            edge30 = b3 - b0;
            return DotCross(edge30, edge10, edge20) / 6.f;
        }

        DYN_FUNC bool getBarycentric(Vec3f & bary,Vec3f p,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3)
        {
            // Compute the vectors relative to V3 of the tetrahedron.
            Vec3f diff[4] = { b0 - b3, b1 - b3, b2 - b3, p - b3 };
            float det = DotCross(diff[0], diff[1], diff[2]);
            if (det < -EPSILON || det > EPSILON) // check if tet is not degenerate
            {
                bary[0] = DotCross(diff[3], diff[1], diff[2]) / det;
                bary[1] = DotCross(diff[3], diff[2], diff[0]) / det;
                bary[2] = DotCross(diff[3], diff[0], diff[1]) / det;
                // outside
                if(REAL_LESS(bary[0], -REAL_ZERO) || REAL_LESS(bary[1], -REAL_ZERO) || REAL_LESS(bary[2], -REAL_ZERO))
                {
                    // printf("[geo] outside (%f %f %f)\n", bary[0], bary[1], bary[2]);
                    return false;
                }
                return true;
            }
            return false;
        }
        
        // Compute the intersection of a line and a triangle in 2D [return num of intersection: 0, 1, 2]
        DYN_FUNC int getIntersection(float& t0, float& t1,
                Vec2f a0, Vec2f a1,
                Vec2f b0, Vec2f b1, Vec2f b2)
        {
            float const zero = 0.f;
            int res = 0;
            Vec3f s( zero, zero, zero );
            int numPositive = 0, numNegative = 0, numZero = 0;
            Vec2f v[3] = { b0, b1, b2 };
            Vec2f diff[3] = { b0 - a0, b1 - a0, b2 - a0 };
            Vec2f direction = a1 - a0;
            Vec2f origin = a0;
            for (size_t i = 0; i < 3; ++i)
            {
                s[i] = DotPerp(direction, diff[i]);
                if (s[i] > zero)
                {
                    ++numPositive;
                }
                else if (s[i] < zero)
                {
                    ++numNegative;
                }
                else
                {
                    ++numZero;
                }
            }

            if (numZero == 0 && numPositive > 0 && numNegative > 0)
            {
                // (n,p,z) is (1,2,0) or (2,1,0).
                // result.intersect = true;
                // result.numIntersections = 2;
                res = 2;

                // sign is +1 when (n,p) is (2,1) or -1 when (n,p) is (1,2).
                float sign = (3 > numPositive * 2 ? 1.f : -1.f);
                for (size_t i0 = 1, i1 = 2, i2 = 0; i2 < 3; i0 = i1, i1 = i2++)
                {
                    if (sign * s[i2] > zero)
                    {
                        Vec2f diffVi0P0 = v[i0] - origin;
                        Vec2f diffVi2Vi0 = v[i2] - v[i0];
                        float lambda0 = s[i0] / (s[i0] - s[i2]);
                        Vec2f q0 = diffVi0P0 + lambda0 * diffVi2Vi0;
                        t0 = Dot(direction, q0);
                        Vec2f diffVi1P0 = v[i1] - origin;
                        Vec2f diffVi2Vi1 = v[i2] - v[i1];
                        float lambda1 = s[i1] / (s[i1] - s[i2]);
                        Vec2f q1 = diffVi1P0 + lambda1 * diffVi2Vi1;
                        t1 = Dot(direction, q1);
                        break;
                    }
                }
            }
            else if (numZero == 1)
            {
                // (n,p,z) is (1,1,1), (2,0,1) or (0,2,1).
                for (size_t i0 = 1, i1 = 2, i2 = 0; i2 < 3; i0 = i1, i1 = i2++)
                {
                    if (s[i2] == zero)
                    {
                        Vec2f diffVi2P0 = v[i2] - origin;
                        t0 = Dot(direction, diffVi2P0);
                        if (numPositive == 2 || numNegative == 2)
                        {
                            // (n,p,z) is (2,0,1) or (0,2,1).
                            res = 1;
                            t1 = t0;
                        }
                        else
                        {
                            // (n,p,z) is (1,1,1).
                            res = 2;
                            Vec2f diffVi0P0 = v[i0] - origin;
                            Vec2f diffVi1Vi0 = v[i1] - v[i0];
                            float lambda0 = s[i0] / (s[i0] - s[i1]);
                            Vec2f q = diffVi0P0 + lambda0 * diffVi1Vi0;
                            t1 = Dot(direction, q);
                        }
                        break;
                    }
                }
            }
            else if (numZero == 2)
            {
                // (n,p,z) is (1,0,2) or (0,1,2).
                res = 2;
                for (size_t i0 = 1, i1 = 2, i2 = 0; i2 < 3; i0 = i1, i1 = i2++)
                {
                    if (s[i2] != zero)
                    {
                        Vec2f diffVi0P0 = v[i0] - origin;
                        t0 = Dot(direction, diffVi0P0);
                        Vec2f diffVi1P0 = v[i1] - origin;
                        t1 = Dot(direction, diffVi1P0);
                        break;
                    }
                }
            }
            // else: (n,p,z) is (3,0,0), (0,3,0) or (0,0,3). The constructor
            // for Result initializes all members to zero, so no additional
            // assignments are needed for 'result'.

            if (res > 0)
            {
                float directionSqrLength = Dot(direction, direction);
                t0 /= directionSqrLength;
                t1 /= directionSqrLength;
                if (t0 > t1)
                {
                    Swap(t0, t1);
                }
            }
            return res;
        }

        // Compute the intersection of two triangle [return num of intersection: 0, 1, 2]
        DYN_FUNC int getIntersection(Vec3f& p0, Vec3f& p1,
                        Vec3f a0, Vec3f a1, Vec3f a2,
                        Vec3f b0, Vec3f b1, Vec3f b2)
        {
            int res = 0;
            int seg_res = 0;
            Vec3f q[2];

            // face A
            Vec3f nA = Cross(a1 - a0, a2 - a0); nA.normalize();
            Vec3f nB = Cross(b1 - b0, b2 - b0); nB.normalize();
            Vec3f oA = a0;

            // if two triangle are coplanar, 
            // whose penetration deep is 0, return 0
            // if two triangle are parallel, return 0
            if (!REAL_LESS(fabs(Dot(nA, nB)), 1.f))
            {
                return 0;
            }

            // check the intersection of three segments and one plane
            float d0 = Dot(nA, b0 - oA);
            float d1 = Dot(nA, b1 - oA);
            float d2 = Dot(nA, b2 - oA);

            int s0 = REAL_EQUAL(d0, 0.f) ? 0 : (REAL_LESS(d0, 0.f) ? -1 : 1);
            int s1 = REAL_EQUAL(d1, 0.f) ? 0 : (REAL_LESS(d1, 0.f) ? -1 : 1);
            int s2 = REAL_EQUAL(d2, 0.f) ? 0 : (REAL_LESS(d2, 0.f) ? -1 : 1);

            // two triangle are coplanar
            if (s0 == 0 && s1 == 0 && s2 == 0) return 0;

            // check if three points in plane
            if (s0 == 0) q[seg_res++] = b0;
            if (s1 == 0) q[seg_res++] = b1;
            if (s2 == 0) q[seg_res++] = b2;

            // check if three segments intersect with plane A
            if (s0 * s1 < 0 && (!REAL_EQUAL(d0 - d1, 0.f))) q[seg_res++] = b0 + (b1 - b0) * d0 / (d0 - d1);
            if (s1 * s2 < 0 && (!REAL_EQUAL(d1 - d2, 0.f))) q[seg_res++] = b1 + (b2 - b1) * d1 / (d1 - d2);
            if (s0 * s2 < 0 && (!REAL_EQUAL(d2 - d0, 0.f))) q[seg_res++] = b2 + (b0 - b2) * d2 / (d2 - d0);

            if (seg_res > 2) printf("[geo] Error: res > 2 %d %d %d %f %f %f\n", s0, s1, s2, d0, d1, d2);
            for (int _ = 0; _ < seg_res; ++_)
                if (q[_] != q[_]) printf("[geo] Error in nan\n");
            // check if one segments(q[0], q[1]) intersect with triangle A
            if (seg_res == 2)
            {
                // printf("[geo] seg Polygon((%f, %f, %f), (%f, %f, %f))\n", q[0][0] * 10, q[0][1] * 10, q[0][2] * 10, q[1][0] * 10, q[1][1] * 10, q[1][2] * 10);
                // 3D -> 2D (Project onto the one of xy- xz- yz- plane)
                int maxIndex = 0;
                float cmax = std::fabs(nA[0]);
                float cvalue = std::fabs(nA[1]);
                if (cvalue > cmax)
                {
                    maxIndex = 1;
                    cmax = cvalue;
                }
                cvalue = std::fabs(nA[2]);
                if (cvalue > cmax)
                {
                    maxIndex = 2;
                }

                Vec3u lookup;
                if (maxIndex == 0)
                {
                    // Project onto the yz-plane.
                    lookup = { 1, 2, 0 };
                }
                else if (maxIndex == 1)
                {
                    // Project onto the xz-plane.
                    lookup = { 0, 2, 1 };
                }
                else // maxIndex = 2
                {
                    // Project onto the xy-plane.
                    lookup = { 0, 1, 2 };
                }

                // Project
                Vec2f a2d[3];
                Vec3f ao[3] = {a0 - oA, a1 - oA, a2 - oA};
                for (size_t i = 0; i < 3; ++i)
                {
                    a2d[i][0] = ao[i][lookup[0]];
                    a2d[i][1] = ao[i][lookup[1]];
                }

                Vec2f q2d[2];
                Vec3f qo[2] ={q[0] - oA, q[1] - oA};
                for (size_t i = 0; i < 2; ++i)
                {
                    q2d[i][0] = qo[i][lookup[0]];
                    q2d[i][1] = qo[i][lookup[1]];
                }

                float t0 = 0.f, t1 = 0.f;
                int res2d = getIntersection(t0, t1, q2d[0], q2d[1], a2d[0], a2d[1], a2d[2]);
                if (res2d == 1)
                {
                    if (REAL_LESS(t0, 1.f) && REAL_GREAT(t0, 0.f))
                    {
                        res = 1;
                        p0 = q[0] + (q[1] - q[0]) * t0;
                    }
                }
                if (res2d == 2)
                {
                    // [0, 1] overlap [t0, t1]
                    if (REAL_LESS(t1, 0.f) || REAL_GREAT(t0, 1.f))
                        return 0;
                    if (REAL_LESS(t0, 0.f)) t0 = 0.f;
                    if (REAL_GREAT(t1, 1.f)) t1 = 1.f;
                    res = 2;
                    p0 = q[0] + (q[1] - q[0]) * t0;
                    p1 = q[0] + (q[1] - q[0]) * t1;
                }
            } 
            // check if one intersection point (q[0]) inside triangle A
            else if (seg_res == 1) 
            {
                // printf("[geo] p Point(%f, %f, %f)\n", q[0][0], q[0][1], q[0][2]);

                int numPositive = 0, numNegative = 0, numZero = 0;
                float s[3] = {Dot(a0 - q[0], a1 - q[0]), Dot(a1 - q[0], a2 - q[0]), Dot(a2 - q[0], a0 - q[0])};
                for (size_t i = 0; i < 3; ++i)
                {
                    if (REAL_GREAT(s[i], 0.f))
                    {
                        ++numPositive;
                    }
                    else if (REAL_LESS(s[i], 0.f))
                    {
                        ++numNegative;
                    }
                    else
                    {
                        ++numZero;
                    }
                }
                if (!(numPositive > 0 && numNegative > 0))
                {
                    res = 1;
                    p0 = q[0];
                }
            }
            return res;
        }

        // get the area of a triangle in a tetrahedron
        DYN_FUNC float getTriBoxAreaInTet(Vec3f a0, Vec3f a1, Vec3f a2,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3)
        {
            float res = 0.f;

            // Intersection with Plane A -> <= 4 point Convex Polygon A
            Vec3f nA = (a1 - a0).cross(a2- a0); // face normal
            nA.normalize();
            Vec3f oA = a0;
            Vec3f t[4] = {b0, b1, b2, b3};
            Vec3f p[4];
            int p_num = 0;

            float d[4]; int s[4];
            for (int i = 0; i < 4; i++) {
                d[i] = Dot(nA, t[i] - oA);
                s[i] = REAL_EQUAL(d[i], 0.f) ? 0 : (REAL_LESS(d[i], 0.f) ? -1 : 1);
            }
            for (int i = 0; i < 4; i++)
            {
                if(s[i] == 0) p[p_num++] = t[i];
                else
                {
                    for (int j = i + 1; j < 4; j++)
                    if (s[i] * s[j] < 0) p[p_num++] = t[j] + (t[i] - t[j]) * d[j] / (d[j] - d[i]);
                }
            }
            if (p_num > 4) printf("[geo] Error: p_num > 4\n");
            // set order.
            if (p_num == 4)
            {
                bool s0 = Sign(nA, p[0], p[1], p[2]);
                bool s1 = Sign(nA, p[0], p[1], p[3]);
                if(s1 != s0) Swap(p[0], p[3]);
                else{
                    bool s2 = Sign(nA, p[0], p[2], p[3]);
                    if(s2 != s0) Swap(p[2], p[3]);
                }
                // if(Sign(nA, p[0], p[1], p[2]) != Sign(nA, p[0], p[2], p[3])) printf("[geo] Error in p order\n");
            }

            // Check if Triangle A is overlap with Polygon A
            Vec3f a[3] = {a0, a1, a2};
            if (p_num <= 2) res = 0.f;
            else {
                res = getOverLapBoxAreaInPoly2D(nA, 3, a, p_num, p);
            }
            if (REAL_LESS(res, 0.f))
            {
                res = 0.f;
                // printf("[geo] Error: res < 0\n");
            }
            return res;
        }

        inline DYN_FUNC Vec2f projectWithParall(Vec3f p, Vec3f a0, Vec3f a1, Vec3f a2, Vec3f a3)
        {
            Vec3f a = a0;
            Vec3f d0 = a1 - a0;
            Vec3f d1 = a3 - a2;

            float A = Dot(d0, d0);
            float B = Dot(d0, d1);
            float C = Dot(d1, d1);
            float D0 = Dot(d0, p);
            float D1 = Dot(d1, p);
            
            float delta = A * C - B * B;
			if (REAL_EQUAL(delta, 0.f)) return Vec2f(0.f,0.f);
			float u = (C * D0 - B * D1) / delta;
			float v = (A * D1 - B * D0) / delta;
            return Vec2f(u, v);
        }


        inline DYN_FUNC int intrSegWithPlane(Vec3f* q,
                Vec3f oA, Vec3f nA,
                Vec3f b0, Vec3f b1)
        {
            int res = 0;
            // check the intersection of one segments and one plane
            float d0 = Dot(nA, b0 - oA);
            float d1 = Dot(nA, b1 - oA);

            int s0 = REAL_EQUAL(d0, 0.f) ? 0 : (REAL_LESS(d0, 0.f) ? -1 : 1);
            int s1 = REAL_EQUAL(d1, 0.f) ? 0 : (REAL_LESS(d1, 0.f) ? -1 : 1);

            // check if two points in plane
            if (s0 == 0) q[res++] = b0;
            if (s1 == 0) q[res++] = b1;

            // check if one segments intersect with plane A
            if (s0 * s1 < 0 && (!REAL_EQUAL(d0 - d1, 0.f))) q[res++] = b0 + (b1 - b0) * d0 / (d0 - d1);

            assert(res <= 2);
            return res;
        }

        inline DYN_FUNC int intrTriWithPlane(Vec3f* q,
                Vec3f oA, Vec3f nA,
                Vec3f b0, Vec3f b1, Vec3f b2)
        {
            int res = 0;
            float d0 = Dot(nA, b0 - oA);
            float d1 = Dot(nA, b1 - oA);
            float d2 = Dot(nA, b2 - oA);

            int s0 = REAL_EQUAL(d0, 0.f) ? 0 : (REAL_LESS(d0, 0.f) ? -1 : 1);
            int s1 = REAL_EQUAL(d1, 0.f) ? 0 : (REAL_LESS(d1, 0.f) ? -1 : 1);
            int s2 = REAL_EQUAL(d2, 0.f) ? 0 : (REAL_LESS(d2, 0.f) ? -1 : 1);

            // check if three points in plane
            if (s0 == 0) q[res++] = b0;
            if (s1 == 0) q[res++] = b1;
            if (s2 == 0) q[res++] = b2;

            // check if three segments intersect with plane A
            if (s0 * s1 < 0 && (!REAL_EQUAL(d0 - d1, 0.f))) q[res++] = b0 + (b1 - b0) * d0 / (d0 - d1);
            if (s1 * s2 < 0 && (!REAL_EQUAL(d1 - d2, 0.f))) q[res++] = b1 + (b2 - b1) * d1 / (d1 - d2);
            if (s0 * s2 < 0 && (!REAL_EQUAL(d2 - d0, 0.f))) q[res++] = b2 + (b0 - b2) * d2 / (d2 - d0);

            assert(res <= 3);
            return res;
        }

        inline DYN_FUNC int intrTetWithPlane(Vec3f* q,
                Vec3f oA, Vec3f nA,
                Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3)
        {
            int res = 0;

            float d0 = Dot(nA, b0 - oA);
            float d1 = Dot(nA, b1 - oA);
            float d2 = Dot(nA, b2 - oA);
            float d3 = Dot(nA, b3 - oA);

            int s0 = REAL_EQUAL(d0, 0.f) ? 0 : (REAL_LESS(d0, 0.f) ? -1 : 1);
            int s1 = REAL_EQUAL(d1, 0.f) ? 0 : (REAL_LESS(d1, 0.f) ? -1 : 1);
            int s2 = REAL_EQUAL(d2, 0.f) ? 0 : (REAL_LESS(d2, 0.f) ? -1 : 1);
            int s3 = REAL_EQUAL(d3, 0.f) ? 0 : (REAL_LESS(d3, 0.f) ? -1 : 1);

            // check if four points in plane
            if (s0 == 0) q[res++] = b0;
            if (s1 == 0) q[res++] = b1;
            if (s2 == 0) q[res++] = b2;
            if (s3 == 0) q[res++] = b3;

            // check if six segments intersect with plane A
            if (s0 * s1 < 0 && (!REAL_EQUAL(d0 - d1, 0.f))) q[res++] = b0 + (b1 - b0) * d0 / (d0 - d1);
            if (s0 * s2 < 0 && (!REAL_EQUAL(d0 - d2, 0.f))) q[res++] = b0 + (b2 - b0) * d0 / (d0 - d2);
            if (s0 * s3 < 0 && (!REAL_EQUAL(d0 - d3, 0.f))) q[res++] = b0 + (b3 - b0) * d0 / (d0 - d3);
            if (s1 * s2 < 0 && (!REAL_EQUAL(d1 - d2, 0.f))) q[res++] = b1 + (b2 - b1) * d1 / (d1 - d2);
            if (s1 * s3 < 0 && (!REAL_EQUAL(d1 - d3, 0.f))) q[res++] = b1 + (b3 - b1) * d1 / (d1 - d3);
            if (s2 * s3 < 0 && (!REAL_EQUAL(d2 - d3, 0.f))) q[res++] = b2 + (b3 - b2) * d2 / (d2 - d3);
            
            assert(res <= 4);
            if (res == 4)
            {
                Vec3f q01 = q[1] - q[0];
                Vec3f q02 = q[2] - q[0];
                Vec3f q03 = q[3] - q[0];
                if (REAL_GREAT(Dot(q01.cross(q02), q01.cross(q03)), 0.f)) Swap(q[2], q[3]);
            }
            return res;
        }

        //    6-------7
        //   /|      /|
        //  / |     / |
        // 4------5   |           
        // |  2 ---|--3         
        // | /     | /          W V
        // 0-------1            |/_U
        inline DYN_FUNC int intrBoxWithPlane(Vec3f* q,
                        Vec3f oA, Vec3f nA,
                        Vec3f center, Vec3f halfU, Vec3f halfV, Vec3f halfW)
        {
            int res = 0;
            Vec3f b0 = center - halfU - halfV - halfW;
            Vec3f b1 = center + halfU - halfV - halfW;
            Vec3f b2 = center - halfU + halfV - halfW;
            Vec3f b3 = center + halfU + halfV - halfW;
            Vec3f b4 = center - halfU - halfV + halfW;
            Vec3f b5 = center + halfU - halfV + halfW;
            Vec3f b6 = center - halfU + halfV + halfW;
            Vec3f b7 = center + halfU + halfV + halfW;
            
            float d0 = Dot(nA, b0 - oA);
            float d1 = Dot(nA, b1 - oA);
            float d2 = Dot(nA, b2 - oA);
            float d3 = Dot(nA, b3 - oA);
            float d4 = Dot(nA, b4 - oA);
            float d5 = Dot(nA, b5 - oA);
            float d6 = Dot(nA, b6 - oA);
            float d7 = Dot(nA, b7 - oA);

            int s0 = REAL_EQUAL(d0, 0.f) ? 0 : (REAL_LESS(d0, 0.f) ? -1 : 1);
            int s1 = REAL_EQUAL(d1, 0.f) ? 0 : (REAL_LESS(d1, 0.f) ? -1 : 1);
            int s2 = REAL_EQUAL(d2, 0.f) ? 0 : (REAL_LESS(d2, 0.f) ? -1 : 1);
            int s3 = REAL_EQUAL(d3, 0.f) ? 0 : (REAL_LESS(d3, 0.f) ? -1 : 1);
            int s4 = REAL_EQUAL(d4, 0.f) ? 0 : (REAL_LESS(d4, 0.f) ? -1 : 1);
            int s5 = REAL_EQUAL(d5, 0.f) ? 0 : (REAL_LESS(d5, 0.f) ? -1 : 1);
            int s6 = REAL_EQUAL(d6, 0.f) ? 0 : (REAL_LESS(d6, 0.f) ? -1 : 1);
            int s7 = REAL_EQUAL(d7, 0.f) ? 0 : (REAL_LESS(d7, 0.f) ? -1 : 1);

            // check if eight points in plane
            if (s0 == 0) q[res++] = b0;
            if (s1 == 0) q[res++] = b1;
            if (s2 == 0) q[res++] = b2;
            if (s3 == 0) q[res++] = b3;
            if (s4 == 0) q[res++] = b4;
            if (s5 == 0) q[res++] = b5;
            if (s6 == 0) q[res++] = b6;
            if (s7 == 0) q[res++] = b7;

            // check if twelve segments intersect with plane A
            if (s0 * s1 < 0 && (!REAL_EQUAL(d0 - d1, 0.f))) q[res++] = b0 + (b1 - b0) * d0 / (d0 - d1);
            if (s0 * s2 < 0 && (!REAL_EQUAL(d0 - d2, 0.f))) q[res++] = b0 + (b2 - b0) * d0 / (d0 - d2);
            if (s0 * s4 < 0 && (!REAL_EQUAL(d0 - d4, 0.f))) q[res++] = b0 + (b4 - b0) * d0 / (d0 - d4);

            if (s1 * s3 < 0 && (!REAL_EQUAL(d1 - d3, 0.f))) q[res++] = b1 + (b3 - b1) * d1 / (d1 - d3);
            if (s1 * s5 < 0 && (!REAL_EQUAL(d1 - d5, 0.f))) q[res++] = b1 + (b5 - b1) * d1 / (d1 - d5);

            if (s2 * s3 < 0 && (!REAL_EQUAL(d2 - d3, 0.f))) q[res++] = b2 + (b3 - b2) * d2 / (d2 - d3);
            if (s2 * s6 < 0 && (!REAL_EQUAL(d2 - d6, 0.f))) q[res++] = b2 + (b6 - b2) * d2 / (d2 - d6);

            if (s3 * s7 < 0 && (!REAL_EQUAL(d3 - d7, 0.f))) q[res++] = b3 + (b7 - b3) * d3 / (d3 - d7);

            if (s4 * s5 < 0 && (!REAL_EQUAL(d4 - d5, 0.f))) q[res++] = b4 + (b5 - b4) * d4 / (d4 - d5);
            if (s4 * s6 < 0 && (!REAL_EQUAL(d4 - d6, 0.f))) q[res++] = b4 + (b6 - b4) * d4 / (d4 - d6);

            if (s5 * s7 < 0 && (!REAL_EQUAL(d5 - d7, 0.f))) q[res++] = b5 + (b7 - b5) * d5 / (d5 - d7);

            if (s6 * s7 < 0 && (!REAL_EQUAL(d6 - d7, 0.f))) q[res++] = b6 + (b7 - b6) * d6 / (d6 - d7);

            assert(res <= 4);
            if (res == 4)
            {
                Vec3f q01 = q[1] - q[0];
                Vec3f q02 = q[2] - q[0];
                Vec3f q03 = q[3] - q[0];
                if (REAL_GREAT(Dot(q01.cross(q02), q01.cross(q03)), 0.f)) Swap(q[2], q[3]);
            }

            return res;   
        }

        inline DYN_FUNC int intrPolyWithLine(float* t,
                    int n, Vec2f* p,
                    Vec2f a0, Vec2f a1)
        {
            Vec2f oA = a0;
            Vec2f dA = a1 - a0;
            int res = 0;

            assert(n >= 3);

            for (int i = 0; i < n; i++)
            {
                int ni = (i == n - 1) ? 0 : i + 1;
                Vec2f oB = p[i];
                Vec2f dB = p[ni] - oB;
                float tA, tB; // tB \in [0, 1]
                float d = DotPerp(dA, dB);
                if (REAL_EQUAL(d, 0.f)) continue;
                tA = DotPerp(dB, oA - oB) / d;
                // if (REAL_EQUAL(tA, 1.f)) continue;
                tB = DotPerp(dA, oA - oB) / d;
                if (REAL_LESS(tB, 0.f) || REAL_GREAT(tB, 1.f)) continue;
                if(REAL_EQUAL(tB, 1.f))
                {
                    int nni = (ni == n - 1) ? 0 : ni + 1;
                    Vec2f ndB = p[nni] - p[ni];
                    float d2 = DotPerp(dA, ndB);
                    if (REAL_GREAT(d * d2 , 0.f)) continue; // up endpoint
                }
                t[res++] = tA;
            }
            if (res > 2)
            {
                for (int i = 0; i < n; i++)
                {
                    int ni = (i == n - 1) ? 0 : i + 1;
                    Vec2f oB = p[i];
                    Vec2f dB = p[ni] - oB;
                    float tA, tB; // tB \in [0, 1]
                    float d = DotPerp(dA, dB);
                    if (REAL_EQUAL(d, 0.f)) continue;
                    tA = DotPerp(dB, oA - oB) / d;
                    // if (REAL_EQUAL(tA, 1.f)) continue;
                    tB = DotPerp(dA, oA - oB) / d;
                    printf("tA tB :%f %f\n", tA, tB);
                    if (REAL_LESS(tB, 0.f) || REAL_GREAT(tB, 1.f)) continue;
                    if (REAL_EQUAL(tB, 1.f))
                    {
                        int nni = (ni == n - 1) ? 0 : ni + 1;
                        Vec2f ndB = p[nni] - p[ni];
                        float d2 = DotPerp(dA, ndB);
                        printf("dd %f %f\n", d, d2);
                        if (REAL_GREAT(d * d2, 0.f)) continue; // up endpoint
                    }
                }
                for (int i = 0; i < res; ++i)
                {
                    Vec2f b = a0 + t[i] * (a1 - a0);
                    printf(" t %f p: %f %f\n", t[i], b[0], b[1]);
                }
                for (int i = 0; i < n; ++i)
					printf("p: %f %f\n", p[i][0], p[i][1]);
				printf("a0 %f %f\n", a0[0], a0[1]);
                printf("a1 %f %f\n", a1[0], a1[1]);
            }
            assert(res <= 2);
            return res;
        }


        inline DYN_FUNC int intrPolyWithTri(Vec3f* q,
                int n, Vec3f* p,
                Vec3f a0, Vec3f a1, Vec3f a2)
        {
            int res = 0;
            Vec3f nA = Cross(a1 - a0, a2 - a0); nA.normalize();

            int maxIndex = 0;
            float cmax = std::fabs(nA[0]);
            float cvalue = std::fabs(nA[1]);
            if (cvalue > cmax)
            {
                maxIndex = 1;
                cmax = cvalue;
            }
            cvalue = std::fabs(nA[2]);
            if (cvalue > cmax) maxIndex = 2;

            Vec3u lookup;
            if (maxIndex == 0)      lookup = { 1, 2, 0 };// Project onto the yz-plane.
            else if (maxIndex == 1) lookup = { 0, 2, 1 }; // Project onto the xz-plane.
            else                    lookup = { 0, 1, 2 }; // Project onto the xy-plane.
            
            // Proj
            Vec2f a2d[3]; Vec2f p2d[4];
            a2d[0] = Vec2f(a0[lookup[0]], a0[lookup[1]]); 
            a2d[1] = Vec2f(a1[lookup[0]], a1[lookup[1]]);
            a2d[2] = Vec2f(a2[lookup[0]], a2[lookup[1]]);
            for (int i = 0; i < n; ++i)
                p2d[i] = Vec2f(p[i][lookup[0]], p[i][lookup[1]]);
            
            if (n == 1)
            {
                // Check Point inside Rect
                Vec2f center = (a2d[0] + a2d[1] + a2d[2]) / 3.0f;
                float t[2];
                int num_intr = intrPolyWithLine(t, 3, a2d, p2d[0], center);
                int num_inside = 0;
                for (int j = 0; j < num_intr; ++j)
                {
                    if (REAL_GREAT(t[j], 0.f)) num_inside++;
                }
                if (num_inside == 1) q[res++] = p[0];
                return res;
            }

            // Check Polygon Edge Intr with Tri
            for (int i = 0; i < n; ++i)
            {
                int ni = (i == n - 1) ? 0 : i + 1;
                float t[2];
                int num_intr = intrPolyWithLine(t, 3, a2d, p2d[i], p2d[ni]); 
                int num_inside = 0;
                for (int j = 0; j < num_intr; ++j)
                {
                    if (REAL_GREAT(t[j], 0.f))
                    {
                        num_inside++;
                        if (REAL_LESS(t[j], 1.f) )
                            q[res++] = p[i] + (p[ni] - p[i]) * t[j];
                    }
                }
                if (num_inside == 1) q[res++] = p[i];
            }

            // Check Tri Point inside Polygon
            if (n > 2)
            {
                for (int i = 0; i < 3; ++i)
                {
                    int ni = (i == 2) ? 0 : i + 1;
                    float t[2];
                    int num_intr = intrPolyWithLine(t, n, p2d, a2d[i], a2d[ni]); 
                    int num_inside = 0;
                    for (int j = 0; j < num_intr; ++j)
                    {
                        if (REAL_GREAT(t[j], 0.f))
                            num_inside++;
                    }
                    if (num_inside == 1) q[res++] = (i == 0) ? a0 : ((i == 1) ? a1 : a2);
                }
            }

            assert(res <= 6);
            return res;
        }

        inline DYN_FUNC int intrPolyWithRect(Vec3f* q,
                int n, Vec3f* p,
                Vec3f a0, Vec3f a1, Vec3f a2, Vec3f a3)
        {
            int res = 0;
            Vec3f nA = Cross(a1 - a0, a2 - a0); nA.normalize();

            int maxIndex = 0;
            float cmax = std::fabs(nA[0]);
            float cvalue = std::fabs(nA[1]);
            if (cvalue > cmax)
            {
                maxIndex = 1;
                cmax = cvalue;
            }
            cvalue = std::fabs(nA[2]);
            if (cvalue > cmax) maxIndex = 2;

            Vec3u lookup;
            if (maxIndex == 0)      lookup = { 1, 2, 0 };// Project onto the yz-plane.
            else if (maxIndex == 1) lookup = { 0, 2, 1 }; // Project onto the xz-plane.
            else                    lookup = { 0, 1, 2 }; // Project onto the xy-plane.
            
            // Proj
            Vec3f a[4]; Vec2f a2d[4]; Vec2f p2d[4];
            a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
            for (int i = 0; i < 4; ++i) a2d[i] = Vec2f(a[i][lookup[0]], a[i][lookup[1]]);
            for (int i = 0; i < n; ++i) p2d[i] = Vec2f(p[i][lookup[0]], p[i][lookup[1]]);
            
            if (n == 1)
            {
				// Check Point inside Rect
                Vec2f center = (a2d[0] + a2d[1] + a2d[2] + a2d[3]) * 0.25f;
                float t[2];
                int num_intr = intrPolyWithLine(t, 4, a2d, p2d[0], center);
                int num_inside = 0;
                for (int j = 0; j < num_intr; ++j)
                {
                    if (REAL_GREAT(t[j], 0.f)) num_inside++;
                }
                if (num_inside == 1) q[res++] = p[0];
                return res;
            }
            // Check Polygon Edge Intr with Rect
            for (int i = 0; i < n; ++i)
            {
                int ni = (i == n - 1) ? 0 : i + 1;
                float t[2];
                int num_intr = intrPolyWithLine(t, 4, a2d, p2d[i], p2d[ni]); 
                int num_inside = 0;
                for (int j = 0; j < num_intr; ++j)
                {
                    if (REAL_GREAT(t[j], 0.f))
                    {
                        num_inside++;
                        if (REAL_LESS(t[j], 1.f) && (ni != 0 || i != 1) ) // same edge
                            q[res++] = p[i] + (p[ni] - p[i]) * t[j];
                    }
                }
                if (num_inside == 1) q[res++] = p[i];
            }

            // Check Tri Point inside Polygon
            if (n > 2)
            {
                for (int i = 0; i < 4; ++i)
                {
                    int ni = (i == 3) ? 0 : i + 1;
                    float t[2];
                    int num_intr = intrPolyWithLine(t, n, p2d, a2d[i], a2d[ni]); 
                    int num_inside = 0;
                    for (int j = 0; j < num_intr; ++j)
                    {
                        if (REAL_GREAT(t[j], 0.f))
                            num_inside++;
                    }
                    if (num_inside == 1) q[res++] = a[i];
                }
            }

            assert(res <= 8);
            return res;
        }

    };
};
