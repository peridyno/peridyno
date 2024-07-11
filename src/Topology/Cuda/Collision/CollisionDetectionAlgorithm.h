#pragma once

#include "CollisionData.h"

namespace dyno
{
    template<typename Real>
    class CollisionDetection
    {
    public:
        using Quat1f = Quat<Real>;
        using Coord3D = Vector<Real, 3>;
        using Matrix3D = SquareMatrix<Real, 3>;
        using Transform3D = Transform<Real, 3>;
        using Tet3D = TTet3D<Real>;
        using Manifold = TManifold<Real>;
        using Sphere3D = TSphere3D<Real>;
        using Segment3D = TSegment3D<Real>;
        using OBox3D = TOrientedBox3D<Real>;
        using Capsule3D = TCapsule3D<Real>;
        using Triangle3D = TTriangle3D<Real>;

        //--------------------------------------------------------------------------------------------------
        // Minkowski Sum + Separating Axis Theorem for Round Primitives
        // Round Primitive	: [Sphere, Capsule(Segment), Triangle, Tetrahedron, Box]

        // MSDF(A, B)		: Minkowski Sum Signed Distance Function, return the depth and normal on B's boundary.
        // request(A, B)	: Generate contact points, depth and normal on B's boundary.
        
        // [Seg - Sphere]
        DYN_FUNC static void MSDF(const Segment3D& segA, const Sphere3D& sphereB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphereA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        
        // [Seg - Seg]
        DYN_FUNC static void MSDF(const Segment3D& segA, const Segment3D& segB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const Segment3D& segB, const Real radiusA, const Real radiusB);

        // [Box - Seg]
        DYN_FUNC static void MSDF(const OBox3D& boxA, const Segment3D& segB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const OBox3D& boxA, const Segment3D& segB, const Real radiusA, const Real radiusB); 
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const OBox3D& boxB, const Real radiusA, const Real radiusB);

        // [Tet- Seg]
        DYN_FUNC static void MSDF(const Tet3D& tetA, const Segment3D& segB, Real& depth, Coord3D& normal, Real& boundary1, Real &boundary2, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Tet3D& tetA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const Tet3D& tetB, const Real radiusA, const Real radiusB); 
        
        // [Tri - Seg]
        DYN_FUNC static void MSDF(const Triangle3D& triA, const Segment3D& segB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Triangle3D& triA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const Triangle3D& triB, const Real radiusA, const Real radiusB);
        





        //--------------------------------------------------------------------------------------------------
        // Resources:
        // https://box2d.googlecode.com/files/GDC2007_ErinCatto.zip
        // https://box2d.googlecode.com/files/Box2D_Lite.zip
        DYN_FUNC static void request(Manifold& m, const OBox3D box0, const OBox3D box1);

        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphere, const OBox3D& box);
        DYN_FUNC static void request(Manifold& m, const OBox3D& box, const Sphere3D& sphere);

        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphere0, const Sphere3D& sphere1);

        DYN_FUNC static void request(Manifold& m, const Tet3D& tet0, const Tet3D& tet1);

        DYN_FUNC static void request(Manifold& m, const Tet3D& tet, const OBox3D& box);
        DYN_FUNC static void request(Manifold& m, const OBox3D& box, const Tet3D& tet);

        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphere, const Tet3D& tet);
        DYN_FUNC static void request(Manifold& m, const Tet3D& tet, const Sphere3D& sphere);

        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphere, const Capsule3D& cap);
        DYN_FUNC static void request(Manifold& m, const Capsule3D& cap, const Sphere3D& sphere);
        
        //=========================================
        DYN_FUNC static void request(Manifold& m, const Capsule3D& cap0, const Capsule3D& cap1);//untested

        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphere, const Triangle3D& tri);//untested
        DYN_FUNC static void request(Manifold& m, const Triangle3D& tri, const Sphere3D& sphere);//untested


        DYN_FUNC static void request(Manifold& m, const Triangle3D& tri, const Capsule3D cap); //untested

        DYN_FUNC static void request(Manifold& m, const Capsule3D& cap, const Tet3D& tet);//untested
        DYN_FUNC static void request(Manifold& m, const Tet3D& tet, const Capsule3D& cap);//untested

        DYN_FUNC static void request(Manifold& m, const Capsule3D& cap, const OBox3D& box);//untested
        DYN_FUNC static void request(Manifold& m, const OBox3D& box, const Capsule3D& cap);//untested

        DYN_FUNC static void request(Manifold& m, const Tet3D& tet, const Triangle3D& tri);

        DYN_FUNC static void request(Manifold& m, const Triangle3D& tri, const Tet3D& tet);

        DYN_FUNC static void request(Manifold& m, const OBox3D& box, const Triangle3D& tri);//unfinished

        //=========================================


    private:
        

    };
}

#include "CollisionDetectionAlgorithm.inl"