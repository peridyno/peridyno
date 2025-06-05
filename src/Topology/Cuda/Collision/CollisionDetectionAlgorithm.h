#pragma once

#include "CollisionData.h"

namespace dyno
{
    enum SeparationType
    {
        CT_POINT = 0,
        CT_EDGE,
        CT_TRIA,
        CT_TRIB,
        CT_RECTA,
        CT_RECTB
    };

    template<typename Real>
    class TSeparationData
    {
        using Tet3D = TTet3D<Real>;
        using Sphere3D = TSphere3D<Real>;
        using Segment3D = TSegment3D<Real>;
        using OBox3D = TOrientedBox3D<Real>;
        using Capsule3D = TCapsule3D<Real>;
        using Triangle3D = TTriangle3D<Real>;
        using Rectangle3D = TRectangle3D<Real>;


        Vector<Real, 3> separation_normal;
        Vector<Real, 3> separation_point[4];
        Real separation_distance = -REAL_INF;
        SeparationType separation_type = CT_POINT;
        int separation_flag = 0;        // [0 A-B, 1 B-A]

    public:
        DYN_FUNC void reverse() { separation_flag = 1 - separation_flag; }

        DYN_FUNC Real depth() { return separation_distance; }
        DYN_FUNC Vector<Real, 3> normal() { return (separation_flag == 1) ? -separation_normal : separation_normal; }
        DYN_FUNC SeparationType type() { return separation_type; }
        DYN_FUNC SeparationType face() { return SeparationType(int(separation_type) ^ separation_flag); } // 0: A, 1: B

        DYN_FUNC Vector<Real, 3> point(int i) { return separation_point[i]; }
        DYN_FUNC Vector<Real, 3> pointA() { return separation_point[0 ^ separation_flag]; }
        DYN_FUNC Vector<Real, 3> pointB() { return separation_point[1 ^ separation_flag]; }
        DYN_FUNC Triangle3D tri() { return Triangle3D(separation_point[0], separation_point[1], separation_point[2]); }
        DYN_FUNC Rectangle3D rect() { return Rectangle3D(separation_point[0], separation_point[1], separation_point[2], Vec2f(separation_point[3][0], separation_point[3][1])); }

        DYN_FUNC void update(SeparationType type, Real BoundaryA, Real BoundaryB, Real Depth, Vec3f N, Vec3f a0, Vec3f a1, Vec3f a2 = Vec3f(0.), Vec3f a3 = Vec3f(0.))
        {
			N = ((BoundaryA < BoundaryB) ^ (isless(Depth, 0.f))) ? N : -N;  // [Question: why -N]
            if (!isless(Depth, 0.f))
            {
                separation_distance = Depth;
                separation_normal = N;
            }
            else if (isgreat(Depth, separation_distance))
            {
                separation_type = type;
                separation_distance = Depth;
                separation_normal = N;
                separation_point[0] = a0;
                separation_point[1] = a1;
                separation_point[2] = a2;
                separation_point[3] = a3;
            }
        }
    };

    template<typename Real>
    class CollisionDetection
    {
    public:
        using Quat1f = Quat<Real>;
        using Coord3D = Vector<Real, 3>;
        using Matrix3D = SquareMatrix<Real, 3>;
        using Transform3D = Transform<Real, 3>;
        using Tet3D = TTet3D<Real>;
        using Sphere3D = TSphere3D<Real>;
        using Segment3D = TSegment3D<Real>;
        using OBox3D = TOrientedBox3D<Real>;
        using Capsule3D = TCapsule3D<Real>;
        using Triangle3D = TTriangle3D<Real>;
        using MedialCone3D = TMedialCone3D<Real>;
        using MedialSlab3D = TMedialSlab3D<Real>;

        using Manifold = TManifold<Real>;
        using SeparationData = TSeparationData<Real>;

        //--------------------------------------------------------------------------------------------------
        // Minkowski Sum + Separating Axis Theorem for Round Primitives
        // Round Primitive	: [Sphere, Capsule(Segment), Triangle, Tetrahedron, Box]

        // MSDF(A, B)		: Minkowski Sum Signed Distance Function, return the depth and normal on B's boundary.
        // request(A, B)	: Generate contact points, depth and normal on B's boundary.
        
        // [Sphere - Sphere]
        DYN_FUNC static void MSDF(SeparationData& sat, const Sphere3D& sphereA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphereA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);

        // [Seg - Sphere]
        DYN_FUNC static void MSDF(SeparationData& sat, const Segment3D& segA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphereA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        
        // [Tri - Sphere]
        DYN_FUNC static void MSDF(SeparationData& sat, const Triangle3D& triA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Triangle3D& triA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphereA, const Triangle3D& triB, const Real radiusA, const Real radiusB);

        // [Tet - Sphere]
        DYN_FUNC static void MSDF(SeparationData& sat, const Tet3D& tetA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Tet3D& tetA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphereA, const Tet3D& tetB, const Real radiusA, const Real radiusB);

        // [Box - Sphere]
		DYN_FUNC static void MSDF(SeparationData& sat, const OBox3D& boxA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const OBox3D& boxA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphereA, const OBox3D& boxB, const Real radiusA, const Real radiusB);


        // [Seg - Seg]
		DYN_FUNC static void MSDF(SeparationData& sat, const Segment3D& segA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const Segment3D& segB, const Real radiusA, const Real radiusB);

        // [Tri - Seg]
        DYN_FUNC static void MSDF(SeparationData& sat, const Triangle3D& triA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Triangle3D& triA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const Triangle3D& triB, const Real radiusA, const Real radiusB);

        // [Tet- Seg]
        DYN_FUNC static void MSDF(SeparationData& sat, const Tet3D& tetA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Tet3D& tetA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const Tet3D& tetB, const Real radiusA, const Real radiusB); 

        // [Box - Seg]
        DYN_FUNC static void MSDF(SeparationData& sat, const OBox3D& boxA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const OBox3D& boxA, const Segment3D& segB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Segment3D& segA, const OBox3D& boxB, const Real radiusA, const Real radiusB);


        // [Tri - Tri]
        DYN_FUNC static void MSDF(SeparationData& sat, const Triangle3D& triA, const Triangle3D& triB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Triangle3D& triA, const Triangle3D& triB, const Real radiusA, const Real radiusB);

        // [Tet - Tri]
        DYN_FUNC static void MSDF(SeparationData& sat, const Tet3D& tetA, const Triangle3D& triB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Tet3D& tetA, const Triangle3D& triB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Triangle3D& triA, const Tet3D& tetB, const Real radiusA, const Real radiusB);

        // [Box - Tri]
        DYN_FUNC static void MSDF(SeparationData& sat, const OBox3D& boxA, const Triangle3D& triB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const OBox3D& boxA, const Triangle3D& triB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Triangle3D& triA, const OBox3D& boxB, const Real radiusA, const Real radiusB);


        // [Tet - Tet]
        DYN_FUNC static void MSDF(SeparationData& sat, const Tet3D& tetA, const Tet3D& tetB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Tet3D& tetA, const Tet3D& tetB, const Real radiusA, const Real radiusB);

        // [Box - Tet]
        DYN_FUNC static void MSDF(SeparationData& sat, const OBox3D& boxA, const Tet3D& tetB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const OBox3D& boxA, const Tet3D& tetB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const Tet3D& tetA, const OBox3D& boxB, const Real radiusA, const Real radiusB);

        
        // [Box - Box]
        DYN_FUNC static void MSDF(SeparationData& sat, const OBox3D& boxA, const OBox3D& boxB, const Real radiusA, const Real radiusB);
        DYN_FUNC static void request(Manifold& m, const OBox3D& boxA, const OBox3D& boxB, const Real radiusA, const Real radiusB);


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
        DYN_FUNC static void request(Manifold& m, const Capsule3D& cap0, const Capsule3D& cap1);

        DYN_FUNC static void request(Manifold& m, const Sphere3D& sphere, const Triangle3D& tri);//untested
        DYN_FUNC static void request(Manifold& m, const Triangle3D& tri, const Sphere3D& sphere);//untested


        DYN_FUNC static void request(Manifold& m, const Triangle3D& tri, const Capsule3D cap); //untested

        DYN_FUNC static void request(Manifold& m, const Capsule3D& cap, const Tet3D& tet);//untested
        DYN_FUNC static void request(Manifold& m, const Tet3D& tet, const Capsule3D& cap);//untested

        DYN_FUNC static void request(Manifold& m, const Capsule3D& cap, const OBox3D& box);
        DYN_FUNC static void request(Manifold& m, const OBox3D& box, const Capsule3D& cap);

        DYN_FUNC static void request(Manifold& m, const Tet3D& tet, const Triangle3D& tri);

        DYN_FUNC static void request(Manifold& m, const Triangle3D& tri, const Tet3D& tet);

        DYN_FUNC static void request(Manifold& m, const OBox3D& box, const Triangle3D& tri);//unfinished

        //=========================================
        DYN_FUNC static void request(Manifold& m, const MedialCone3D& medialcone1, const MedialCone3D& medialcone2); 

        // contact pos : tag = 0: on slab | tag = 1: on sphere
        DYN_FUNC static void request(Manifold& m, const MedialSlab3D& medialslab, const Sphere3D& sphere, int tag);

        DYN_FUNC static void request(Manifold& m, const MedialSlab3D& medialslab, const MedialCone3D& medialcone);

        DYN_FUNC static void request(Manifold& m, const MedialCone3D& medialcone, const MedialSlab3D& medialslab);

        DYN_FUNC static void request(Manifold& m, const MedialSlab3D& medialcone, const MedialSlab3D& medialslab);
    private:
        

    };
}

#include "CollisionDetectionAlgorithm.inl"
