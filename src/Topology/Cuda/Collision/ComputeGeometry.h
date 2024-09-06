/**
 * Copyright 2023 Xukun Luo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "Platform.h"

#include "Matrix.h"
#include "Vector/Vector3D.h"
#include "Vector.h"

namespace dyno
{
    // Refer to https://www.geometrictools.com/index.html
    namespace cgeo
    {
        template<typename T>
        inline DYN_FUNC void Swap(T& a, T& b);

        inline DYN_FUNC float Dot(Vec3f const& U, Vec3f const& V);
        inline DYN_FUNC float Dot(Vec2f const& U, Vec2f const& V);
        inline DYN_FUNC Vec2f Perp(Vec2f const& v);
        inline DYN_FUNC float DotPerp(Vec2f const& v0, Vec2f const& v1);
        // U x V
        inline DYN_FUNC Vec3f Cross(Vec3f const& U, Vec3f const& V);
        // U (V x W)
        inline DYN_FUNC float DotCross(Vec3f const& U, Vec3f const& V, Vec3f const& W);
        inline DYN_FUNC bool isOverLap(float& c0, float& c1, float a0, float a1, float b0, float b1);
        inline DYN_FUNC bool Sign(Vec3f const& n, Vec3f const& p0, Vec3f const& p1, Vec3f const& p2);

        inline DYN_FUNC Vec3f getProjectionVF(Vec3f p,
                            Vec3f a0, Vec3f a1, Vec3f a2);

        inline DYN_FUNC Vec3f getDirectionVF(Vec3f p,
                            Vec3f a0, Vec3f a1, Vec3f a2);

        inline DYN_FUNC float getDistanceVF(Vec3f p,
                            Vec3f a0, Vec3f a1, Vec3f a2);

        inline DYN_FUNC bool isInNarrowBand(Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3,
                        Vec3f a0, Vec3f a1, Vec3f a2, float d);

        // Test for intersection of two Convex Polygon in 2D. (2D Separating Axis)
        inline DYN_FUNC bool isConPolyOverLap2D(Vec3f face_n, int n_a, Vec3f* a,
                        int n_b, Vec3f* b);

        // Test for intersection of a triangle and a tetrahedron.
        inline DYN_FUNC bool isIntrTri2Tet(Vec3f a0, Vec3f a1, Vec3f a2,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3);

        // Test for intersection of a triangle and a tetrahedron that have a shared vertex.
        inline DYN_FUNC bool isIntrTri2Tet(Vec3f a0, Vec3f a1, Vec3f a2,
                    Vec3f b1, Vec3f b2, Vec3f b3);   // b0 = a0

        // Test for containment of a point by a tetrahedron.
        inline DYN_FUNC bool isInTet(Vec3f p,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3);

        // get the box area(approx) of intersection of two Convex Polygon in 2D
        inline DYN_FUNC float getOverLapBoxAreaInPoly2D(Vec3f face_n, int n_a, Vec3f* a,
                        int n_b, Vec3f* b);

        // get the box area(approx) of a triangle in a tetrahedron
        inline DYN_FUNC float getTriBoxAreaInTet(Vec3f a0, Vec3f a1, Vec3f a2,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3);

        // Volume for tetrahedron(b0 b1 b2 b3)
        inline DYN_FUNC float getVolume(Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3);
        
        // Barycentric coordinates(l0l1l2, l3 = 1-l0-l1-l2) for a point p with respect to a tetrahedron (b0 b1 b2 b3)
        inline DYN_FUNC bool getBarycentric(Vec3f& bary, Vec3f p,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3);

        // Compute the intersection of a line and a triangle in 2D [return num of intersection: 0, 1, 2]
        inline DYN_FUNC int getIntersection(float& t0, float& t1,
                Vec2f a0, Vec2f a1,
                Vec2f b0, Vec2f b1, Vec2f b2);

        // Compute the intersection of two triangle [return num of intersection: 0, 1, 2]
        inline DYN_FUNC int getIntersection(Vec3f& p0, Vec3f& p1,
                        Vec3f a0, Vec3f a1, Vec3f a2,
                        Vec3f b0, Vec3f b1, Vec3f b2);


        // Compute Projection of p onto parallelogram
        inline DYN_FUNC Vec2f projectWithParall(Vec3f p, Vec3f a0, Vec3f a1, Vec3f a2, Vec3f a3);


        // Compute the intersection of a plane and a seg [return num of intersection: <= 2]
        inline DYN_FUNC int intrSegWithPlane(Vec3f* q,
                        Vec3f oA, Vec3f nA,
                        Vec3f b0, Vec3f b1);

        // Compute the intersection of a plane and a triangle [return num of intersection: <= 3]
        inline DYN_FUNC int intrTriWithPlane(Vec3f* q,
                        Vec3f oA, Vec3f nA,
                        Vec3f b0, Vec3f b1, Vec3f b2);

        // Compute the intersection of a plane and a tet [return num of intersection: <= 4]
        inline DYN_FUNC int intrTetWithPlane(Vec3f* q,
                        Vec3f oA, Vec3f nA,
                        Vec3f b0, Vec3f b1, Vec3f b2, Vec3f b3);

        // Compute the intersection of a plane and a box [return num of intersection: <= 4]
        inline DYN_FUNC int intrBoxWithPlane(Vec3f* q,
                        Vec3f oA, Vec3f nA,
                        Vec3f center, Vec3f halfU, Vec3f halfV, Vec3f halfW);

        // Compute the intersection of polygon( 3<=n<=4 ) with line on 2D [return num of intersection: <= 2]
        inline DYN_FUNC int intrPolyWithLine(float* t,
                    int n, Vec2f* p,
                    Vec2f a0, Vec2f a1);

        // Compute the intersection of polygon( n<=4 ) with coplanar tri on 3D [return num of intersection: <= 6]
        inline DYN_FUNC int intrPolyWithTri(Vec3f* q,
                        int n, Vec3f* p,
                        Vec3f a0, Vec3f a1, Vec3f a2);

        // Compute the intersection of polygon( n<=4 ) with coplanar rect on 3D [return num of intersection: <= 8]
        inline DYN_FUNC int intrPolyWithRect(Vec3f* q,
                        int n, Vec3f* p,
                        Vec3f a0, Vec3f a1, Vec3f a2, Vec3f a3);
    };
};
#include "ComputeGeometry.inl"