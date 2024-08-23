#include "Platform.h"
#include "Primitive/Primitive3D.h"
#include "Vector/Vector3D.h"
#include "ComputeGeometry.h"

namespace dyno
{

    #define REAL_infinity 1.0e30
    #define	REAL_EQUAL(a,b)  (((a < b + EPSILON) && (a > b - EPSILON)) ? true : false)
    #define REAL_GREAT(a,b) ((a > EPSILON + b)? true: false) 
    #define REAL_LESS(a,b) ((a + EPSILON < b)? true: false)

    struct ClipVertex
    {
        Vector<Real, 3> v;
    };

    template<typename Real>
    DYN_FUNC float fsign(Real v)
    {
        return v < 0 ? -Real(1) : Real(1);
    }

    template<typename Real>
    DYN_FUNC void swapContactPair(TManifold<Real>& m)
    {
        m.normal = -m.normal;

        for (uint i = 0; i < m.contactCount; i++) {
            m.contacts[i].position += m.contacts[i].penetration * m.normal;
        }
    }

    //--------------------------------------------------------------------------------------------------
    // qBoxtoBox
    //--------------------------------------------------------------------------------------------------
    template<typename Real>
    DYN_FUNC inline bool trackFaceAxis(int& axis, Real& sMax, Vector<Real, 3>& axisNormal, int n, Real s, const Vector<Real, 3>& normal)
    {
        if (s > Real(0))
            return true;

        if (s > sMax)
        {
            sMax = s;
            axis = n;
            axisNormal = normal;
        }

        return false;
    }

    //--------------------------------------------------------------------------------------------------
    template<typename Real>
    DYN_FUNC inline bool trackEdgeAxis(int& axis, Real& sMax, Vector<Real, 3>& axisNormal, int n, Real s, const Vector<Real, 3>& normal)
    {
        if (s > Real(0))
            return true;

        Real l = Real(1) / normal.norm();
        s *= l;

        if (s > sMax)
        {
            sMax = s;
            axis = n;
            axisNormal = normal * l;
        }

        return false;
    }
    
    
    /**
     *		Ordering of the edges
     *					   ---2  --- 
     *				   /			 /
     *				  3			    1
     *				 /			   /
     *				  ---   0  --- 
     *					|			 |
     *					10			 9	
     *					|			 |
     *	 				 ---- 4  --- ----->x
     *				|   /       |   /
     *			   11  5		8  7
     *				| /			| /
     *				 / --- 6 --- 
     *             z
     */
    template<typename Real>
    //--------------------------------------------------------------------------------------------------
    DYN_FUNC void computeReferenceEdgesAndBasis(unsigned char* out, SquareMatrix<Real, 3>* basis, Vector<Real, 3>* e, const Vector<Real, 3>& eR, const Transform<Real, 3>& rtx, Vector<Real, 3> n, int axis)
    {
        n = rtx.rotation().transpose()*n;

        if (axis >= 3)
            axis -= 3;

        auto rot_t = rtx.rotation();
        SquareMatrix<Real, 3> outB;

        switch (axis)
        {
        case 0:
            if (n.x > Real(0.0))
            {
                out[0] = 1;
                out[1] = 8;
                out[2] = 7;
                out[3] = 9;

                *e = Vector<Real, 3>(eR.y, eR.z, eR.x);
                //basis->SetRows(rtx.rotation.ey, rtx.rotation.ez, rtx.rotation.ex);
                outB.setCol(0, rot_t.col(1));
                outB.setCol(1, rot_t.col(2));
                outB.setCol(2, rot_t.col(0));
            }
            else
            {
                out[0] = 11;
                out[1] = 3;
                out[2] = 10;
                out[3] = 5;

                *e = Vector<Real, 3>(eR.z, eR.y, eR.x);
                //basis->SetRows(rtx.rotation.ez, rtx.rotation.ey, -rtx.rotation.ex);
                outB.setCol(0, rot_t.col(2));
                outB.setCol(1, rot_t.col(1));
                outB.setCol(2, -rot_t.col(0));
            }
            break;

        case 1:
            if (n.y > Real(0.0))
            {
                out[0] = 0;
                out[1] = 1;
                out[2] = 2;
                out[3] = 3;

                *e = Vector<Real, 3>(eR.z, eR.x, eR.y);
                //basis->SetRows(rtx.rotation.ez, rtx.rotation.ex, rtx.rotation.ey);
                outB.setCol(0, rot_t.col(2));
                outB.setCol(1, rot_t.col(0));
                outB.setCol(2, rot_t.col(1));
            }
            else
            {
                out[0] = 4;
                out[1] = 5;
                out[2] = 6;
                out[3] = 7;

                *e = Vector<Real, 3>(eR.z, eR.x, eR.y);
                //basis->SetRows(rtx.rotation.ez, -rtx.rotation.ex, -rtx.rotation.ey);
                outB.setCol(0, rot_t.col(2));
                outB.setCol(1, -rot_t.col(0));
                outB.setCol(2, -rot_t.col(1));
            }
            break;

        case 2:
            if (n.z > Real(0.0))
            {
                out[0] = 11;
                out[1] = 4;
                out[2] = 8;
                out[3] = 0;

                *e = Vector<Real, 3>(eR.y, eR.x, eR.z);
                //basis->SetRows(-rtx.rotation.ey, rtx.rotation.ex, rtx.rotation.ez);
                outB.setCol(0, -rot_t.col(1));
                outB.setCol(1, rot_t.col(0));
                outB.setCol(2, rot_t.col(2));
            }
            else
            {
                out[0] = 6;
                out[1] = 10;
                out[2] = 2;
                out[3] = 9;

                *e = Vector<Real, 3>(eR.y, eR.x, eR.z);
                //basis->SetRows(-rtx.rotation.ey, -rtx.rotation.ex, -rtx.rotation.ez);
                outB.setCol(0, -rot_t.col(1));
                outB.setCol(1, -rot_t.col(0));
                outB.setCol(2, -rot_t.col(2));
            }
            break;
        }

        *basis = outB;
    }

    //--------------------------------------------------------------------------------------------------
    template<typename Real>
    DYN_FUNC void computeIncidentFace(ClipVertex* out, const Transform<Real, 3>& itx, const Vector<Real, 3>& e, Vector<Real, 3> n)
    {
        n = -itx.rotation().transpose()*n;
        Vector<Real, 3> absN = abs(n);

        if (absN.x > absN.y && absN.x > absN.z)
        {
            if (n.x > Real(0.0))
            {
                out[0].v = Vector<Real, 3>(e.x, e.y, -e.z);
                out[1].v = Vector<Real, 3>(e.x, e.y, e.z);
                out[2].v = Vector<Real, 3>(e.x, -e.y, e.z);
                out[3].v = Vector<Real, 3>(e.x, -e.y, -e.z);
            }
            else
            {
                out[0].v = Vector<Real, 3>(-e.x, -e.y, e.z);
                out[1].v = Vector<Real, 3>(-e.x, e.y, e.z);
                out[2].v = Vector<Real, 3>(-e.x, e.y, -e.z);
                out[3].v = Vector<Real, 3>(-e.x, -e.y, -e.z);
            }
        }
        else if (absN.y > absN.x && absN.y > absN.z)
        {
            if (n.y > Real(0.0))
            {
                out[0].v = Vector<Real, 3>(-e.x, e.y, e.z);
                out[1].v = Vector<Real, 3>(e.x, e.y, e.z);
                out[2].v = Vector<Real, 3>(e.x, e.y, -e.z);
                out[3].v = Vector<Real, 3>(-e.x, e.y, -e.z);
            }
            else
            {
                out[0].v = Vector<Real, 3>(e.x, -e.y, e.z);
                out[1].v = Vector<Real, 3>(-e.x, -e.y, e.z);
                out[2].v = Vector<Real, 3>(-e.x, -e.y, -e.z);
                out[3].v = Vector<Real, 3>(e.x, -e.y, -e.z);
            }
        }
        else
        {
            if (n.z > Real(0.0))
            {
                out[0].v = Vector<Real, 3>(-e.x, e.y, e.z);
                out[1].v = Vector<Real, 3>(-e.x, -e.y, e.z);
                out[2].v = Vector<Real, 3>(e.x, -e.y, e.z);
                out[3].v = Vector<Real, 3>(e.x, e.y, e.z);
            }
            else
            {
                out[0].v = Vector<Real, 3>(e.x, -e.y, -e.z);
                out[1].v = Vector<Real, 3>(-e.x, -e.y, -e.z);
                out[2].v = Vector<Real, 3>(-e.x, e.y, -e.z);
                out[3].v = Vector<Real, 3>(e.x, e.y, -e.z);
            }
        }

        for (int i = 0; i < 4; ++i)
            out[i].v = itx * out[i].v;
    }

    //--------------------------------------------------------------------------------------------------
#define InFront( a ) \
    ((a) < float( 0.0 ))

#define Behind( a ) \
    ((a) >= float( 0.0 ))

#define On( a ) \
    ((a) < float( 0.005 ) && (a) > -float( 0.005 ))

    template<typename Real>
    DYN_FUNC int orthographic(ClipVertex* out, Real sign, Real e, int axis, int clipEdge, ClipVertex* in, int inCount)
    {
        int outCount = 0;
        ClipVertex a = in[inCount - 1];

        for (int i = 0; i < inCount; ++i)
        {
            ClipVertex b = in[i];

            Real da = sign * a.v[axis] - e;
            Real db = sign * b.v[axis] - e;

            ClipVertex cv;

            // B
            if (((InFront(da) && InFront(db)) || On(da) || On(db)))
            {
                out[outCount++] = b;
            }
            // I
            else if (InFront(da) && Behind(db))
            {
                cv.v = a.v + (b.v - a.v) * (da / (da - db));
                out[outCount++] = cv;
            }
            else if (Behind(da) && InFront(db))
            {
                cv.v = a.v + (b.v - a.v) * (da / (da - db));
                out[outCount++] = cv;
                out[outCount++] = b;
            }

            a = b;
        }

        return outCount;
    }

    //--------------------------------------------------------------------------------------------------
    // Resources (also see q3BoxtoBox's resources):
    template<typename Real>
    DYN_FUNC int clip(ClipVertex* outVerts, float* outDepths, const Vector<Real, 3>& rPos, const Vector<Real, 3>& e, unsigned char* clipEdges, const SquareMatrix<Real, 3>& basis, ClipVertex* incident)
    {
        int inCount = 4;
        int outCount;
        ClipVertex in[8];
        ClipVertex out[8];

        for (int i = 0; i < 4; ++i)
            in[i].v = basis.transpose()*(incident[i].v - rPos);

        outCount = orthographic(out, Real(1.0), e.x, 0, clipEdges[0], in, inCount);

        if (outCount == 0)
            return 0;

        inCount = orthographic(in, Real(1.0), e.y, 1, clipEdges[1], out, outCount);

        if (inCount == 0)
            return 0;

        outCount = orthographic(out, Real(-1.0), e.x, 0, clipEdges[2], in, inCount);

        if (outCount == 0)
            return 0;

        inCount = orthographic(in, Real(-1.0), e.y, 1, clipEdges[3], out, outCount);

        // Keep incident vertices behind the reference face
        outCount = 0;
        for (int i = 0; i < inCount; ++i)
        {
            Real d = in[i].v.z - e.z;

            if (d <= Real(0.0))
            {
                outVerts[outCount].v = basis * in[i].v + rPos;
                outDepths[outCount++] = d;
            }
        }

        //assert(outCount <= 8);

        return outCount;
    }

    //--------------------------------------------------------------------------------------------------
    template<typename Real>
    DYN_FUNC inline void edgesContact(Vector<Real, 3>& CA, Vector<Real, 3>& CB, const Vector<Real, 3>& PA, const Vector<Real, 3>& QA, const Vector<Real, 3>& PB, const Vector<Real, 3>& QB)
    {
        Vector<Real, 3> DA = QA - PA;
        Vector<Real, 3> DB = QB - PB;
        Vector<Real, 3> r = PA - PB;
        Real a = DA.dot(DA);
        Real e = DB.dot(DB);
        Real f = DB.dot(r);
        Real c = DA.dot(r);

        Real b = DA.dot(DB);
        Real denom = a * e - b * b;

        Real TA = (b * f - c * e) / denom;
        Real TB = (b * TA + f) / e;

        CA = PA + DA * TA;
        CB = PB + DB * TB;
    }

    // ---------------------------------------- [   MSDF  ] ----------------------------------------
    template<typename Real>
    DYN_FUNC inline void checkSignedDistance(
        Real lowerBoundaryA,
        Real upperBoundaryA, // A
        Real lowerBoundaryB,
        Real upperBoundaryB, // B
        Real& intersectionDistance, // +:outside  -:inside
        Real& boundaryA,	// A
        Real& boundaryB		// B
    )
    {
        if (!((lowerBoundaryA > upperBoundaryB) || (lowerBoundaryB > upperBoundaryA)))
        {
            // boundaryB < boundaryA :B (->) [default: N]
            // boundaryA < boundaryB :B (<-) [-N]
            if (lowerBoundaryA < lowerBoundaryB)
            {
                if (upperBoundaryA > upperBoundaryB)
                {
                    //     |---B---|
                    //   |-----A-----|
                    if (upperBoundaryB - lowerBoundaryA > upperBoundaryA - lowerBoundaryB)
                    {
                        //      |---B---|(->)
                        //   |-----A-----|
                        boundaryA = upperBoundaryA;
                        boundaryB = lowerBoundaryB;
                        intersectionDistance = - (upperBoundaryA - lowerBoundaryB);
                    }
                    else
                    {
                        // (<-)|---B---|
                        //    |-----A-----|
                        boundaryA = lowerBoundaryA;
                        boundaryB = upperBoundaryB;
                        intersectionDistance = - (upperBoundaryB - lowerBoundaryA);
                    }
                }
                else
                {
                    //	        |---B---|(->)
                    //   |----A----|
                    boundaryA = upperBoundaryA;
                    boundaryB = lowerBoundaryB;
                    intersectionDistance = - (upperBoundaryA - lowerBoundaryB);
                }
            }
            else
            {
                if (upperBoundaryA > upperBoundaryB)
                {
                    //	(<-)|---B---|
                    //            |----A----|
                    boundaryA = lowerBoundaryA;
                    boundaryB = upperBoundaryB;
                    intersectionDistance = - (upperBoundaryB - lowerBoundaryA);
                }
                else
                {
                    //     |-----B------|
                    //        |---A---|
                    //intersectionDistance = upperBoundaryA - lowerBoundaryA;
                    if (upperBoundaryB - lowerBoundaryA > upperBoundaryA - lowerBoundaryB)
                    {
                        //	   |-----B-----|(->)
                        //      |---A---|
                        boundaryA = upperBoundaryA;
                        boundaryB = lowerBoundaryB;
                        intersectionDistance = - (upperBoundaryA - lowerBoundaryB);
                    }
                    else
                    {
                        //	   (<-)|------B------|
                        //              |---A---|
                        boundaryA = lowerBoundaryA;
                        boundaryB = upperBoundaryB;
                        intersectionDistance = - (upperBoundaryB - lowerBoundaryA);
                    }
                }
            }
        }
        else
        {
            // boundaryA < boundaryB :B (->) [default: N]
            // boundaryB < boundaryA :B (<-) [-N]
            // |---A---| (->) |---B---|
            if (upperBoundaryA < lowerBoundaryB)
            {
                boundaryA = upperBoundaryA;
                boundaryB = lowerBoundaryB;
                intersectionDistance = (lowerBoundaryB - upperBoundaryA);
            }
            // |---B---| (<-) |---A---|
            if (upperBoundaryB < lowerBoundaryA)
            {
                boundaryA = lowerBoundaryA;
                boundaryB = upperBoundaryB;
                intersectionDistance = (lowerBoundaryA - upperBoundaryB);
            }
        }
    }

    template<typename Real>
    DYN_FUNC inline bool checkPointInBoundary(const Vec3f& p, const Vec3f& N, const Real& b, const Real& r)
    {
        Real c = p.dot(N);
        return (!REAL_GREAT(c, b + r) && !REAL_LESS(c, b - r));
    };

    template<typename Real>
    DYN_FUNC inline void updateSDF(
        Real& boundaryA,
        Real& boundaryB,
        Real& depth,
        Vec3f& normal,
        Real currentBoundaryA,
        Real currentBoundaryB,
        Real currentDepth,
        Vec3f currentN
    )
    {
        // SDF Calculate on Convex Hull
        // - : minimum distance
        // + : maximum distance
        currentN = ( (currentBoundaryB < currentBoundaryA) ^ (REAL_LESS(currentDepth, 0)) ) ? -currentN : currentN;
        if (REAL_LESS(currentDepth, 0) && REAL_GREAT(currentDepth , depth))
        {
            depth = currentDepth;
            normal = currentN;
            boundaryA = currentBoundaryA;
            boundaryB = currentBoundaryB;
        }
    }

    template<typename Real, typename ShapeA, typename ShapeB>
    DYN_FUNC inline void checkSignedDistanceAxis(Real& intersectionDistance, Real& BoundaryA, Real& BoundaryB, const Vec3f axisNormal, ShapeA& shapeA, ShapeB& shapeB, const Real radiusA, const Real radiusB)
    {
        // Contact normal on B
        Real lowerBoundaryA, upperBoundaryA, lowerBoundaryB, upperBoundaryB;

        // projection to axis
        projectOnAxis(lowerBoundaryA, upperBoundaryA, axisNormal, shapeA, radiusA);
        projectOnAxis(lowerBoundaryB, upperBoundaryB, axisNormal, shapeB, radiusB);

        checkSignedDistance(lowerBoundaryA, upperBoundaryA, lowerBoundaryB, upperBoundaryB, intersectionDistance, BoundaryA, BoundaryB);
    }


    template<typename Real, typename ShapeA, typename ShapeB>
    DYN_FUNC inline void checkAxisPoint(
        TSeparationData<Real>& sat,
        ShapeA& shapeA,
        ShapeB& shapeB,
        const Real radiusA,
        const Real radiusB,
        Vec3f pA,
        Vec3f pB,
        const Real rA = 0.f,
        const Real rB = 0.f // for Sphere
        )
    {
        Vec3f N = pB - pA;
        Real D = 0;
        Real bA, bB;
        if (N.norm() > EPSILON) N /= N.norm(); else return;
        checkSignedDistanceAxis(D, bA, bB, N, shapeA, shapeB, radiusA, radiusB);

        if (!checkPointInBoundary(pA, N, bA, radiusA + rA) || !checkPointInBoundary(pB, N, bB, radiusB + rB)) return;
        sat.update(SeparationType::CT_POINT, bA, bB, D, N, pA, pB);
    }


    template<typename Real, typename ShapeA, typename ShapeB>
    DYN_FUNC inline void checkAxisEdge(
        TSeparationData<Real>& sat,
        ShapeA& shapeA,
        ShapeB& shapeB,
        const Real radiusA,
        const Real radiusB,
        Segment3D edgeA,
        Segment3D edgeB
    )
    {
        Vec3f dirA = edgeA.direction();
        Vec3f dirB = edgeB.direction();
        Segment3D proj = edgeA.proximity(edgeB);
		Vec3f N = dirA.cross(dirB);
        Real D = 0;
        Real bA, bB;
        if (N.norm() > EPSILON) N /= N.norm(); else return;
        checkSignedDistanceAxis(D, bA, bB, N, shapeA, shapeB, radiusA, radiusB);

        if (!checkPointInBoundary(proj.v0, N, bA, radiusA) || !checkPointInBoundary(proj.v1, N, bB, radiusB)) return;
        sat.update(SeparationType::CT_EDGE, bA, bB, D, N, proj.v0, proj.v1);
    }

    template<typename Real, typename ShapeA, typename ShapeB>
    DYN_FUNC inline void checkAxisTri(
        TSeparationData<Real>& sat,
        ShapeA& shapeA,
        ShapeB& shapeB,
        const Real radiusA,
        const Real radiusB,
		Triangle3D tri,
        SeparationType type // [faceA or faceB]
    )
	{
		Vec3f N = tri.normal();
		Real D = 0;
		Real bA, bB;
		if (N.norm() > EPSILON) N /= N.norm(); else return;
		checkSignedDistanceAxis(D, bA, bB, N, shapeA, shapeB, radiusA, radiusB);
        Real bb = (type == CT_TRIA) ? bA : bB;
        Real rr = (type == CT_TRIA) ? radiusA : radiusB;
        for (int i = 0; i < 3; ++i)
            if (!checkPointInBoundary(tri.v[i], N, bb, rr)) return;
        sat.update(type, bA, bB, D, N, tri.v[0], tri.v[1], tri.v[2]);
    }

    template<typename Real, typename ShapeA, typename ShapeB>
    DYN_FUNC inline void checkAxisRect(
        TSeparationData<Real>& sat,
        ShapeA& shapeA,
        ShapeB& shapeB,
        const Real radiusA,
        const Real radiusB,
        Rectangle3D rect,
        SeparationType type
    )
    {
        Vec3f N = rect.normal();
        Real D = 0;
        Real bA, bB;
        if (N.norm() > EPSILON) N /= N.norm(); else return;
        checkSignedDistanceAxis(D, bA, bB, N, shapeA, shapeB, radiusA, radiusB);
        Real bb = (type == CT_RECTA) ? bA : bB;
        Real rr = (type == CT_RECTA) ? radiusA : radiusB;
        for (int i = 0; i < 4; ++i) 
            if (!checkPointInBoundary(rect.vertex(i).origin, N, bb, rr)) return;
        sat.update(type, bA, bB, D, N, rect.center, rect.axis[0], rect.axis[1], Vec3f(rect.extent[0], rect.extent[1], 0.f));
    }

    // ---------------------------------------- [ Sphere ] ----------------------------------------
    template<typename Real>
    DYN_FUNC inline void projectOnAxis(
        Real& lowerBoundary,
        Real& upperBoundary,
        const Vec3f axisNormal,
        Sphere3D sphere,
        const Real radius
    )
    {
        lowerBoundary = upperBoundary = sphere.center.dot(axisNormal);

        lowerBoundary -= radius + sphere.radius;
        upperBoundary += radius + sphere.radius;
    }

    template<typename Real>
    DYN_FUNC inline void setupContactOnSphere(
        TManifold<Real>& m,
        TSeparationData<Real>& sat,
        const TSphere3D<Real>& sphereB,
        const Real radiusA,
        const Real radiusB)
    {
        Vec3f contactPoint = sat.pointB() - m.normal * (sphereB.radius);
        m.pushContact(contactPoint, sat.depth());
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithTri(
        Vector<Real, 3>* q,
        const TTriangle3D<Real>& triA,
        const TSphere3D<Real>& sphereB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        return 0;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithRect(
        Vector<Real, 3>* q,
        const TRectangle3D<Real>& rectA,
        const TSphere3D<Real>& sphereB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        return 0;
    }

    // ---------------------------------------- [ Segment ] ----------------------------------------
    template<typename Real>
    DYN_FUNC inline void projectOnAxis(
        Real& lowerBoundary,
        Real& upperBoundary,
        const Vec3f axisNormal,
        Segment3D seg,
        const Real radius
    )
    {
        Real t = seg.v0.dot(axisNormal); lowerBoundary = upperBoundary = t;
        t = seg.v1.dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

        lowerBoundary -= radius;
        upperBoundary += radius;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithTri(
        Vector<Real, 3>* q,
        const TTriangle3D<Real>& triA,
        const TSegment3D<Real>& segB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        Triangle3D triT = Triangle3D(triA.v[0] + transA, triA.v[1] + transA, triA.v[2] + transA);
        Segment3D segT = Segment3D(segB.v0 + transB, segB.v1 + transB);
        
        int num = 0;
		Vec3f oA = triT.v[0];
		Vec3f nA = triT.normal();
        Vec3f poly[2];

		num = cgeo::intrSegWithPlane(poly, oA, nA, segT.v0, segT.v1);
        if (num > 0) num = cgeo::intrPolyWithTri(q, num, poly, triT.v[0], triT.v[1], triT.v[2]);

        return num;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithRect(
        Vector<Real, 3>* q,
        const TRectangle3D<Real>& rectA,
        const TSegment3D<Real>& segB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        Rectangle3D rectT = rectA; rectT.center+= transA;
        Segment3D segT = Segment3D(segB.v0 + transB, segB.v1 + transB);

        int num = 0;
        Vec3f oA = rectT.center;
        Vec3f nA = rectT.normal();
        Vec3f poly[2];

        num = cgeo::intrSegWithPlane(poly, oA, nA, segT.v0, segT.v1);
        if(num > 0) num = cgeo::intrPolyWithRect(q, num, poly, rectT.vertex(0).origin, rectT.vertex(1).origin, rectT.vertex(2).origin, rectT.vertex(3).origin);
        return num;
    }

    template<typename Real>
    DYN_FUNC inline void setupContactOnSeg(
        TManifold<Real>& m,
        TSeparationData<Real>& sat,
        const TSegment3D<Real>& segB,
        const Real radiusA,
        const Real radiusB)
    {
		Real depth = sat.depth();
        if (sat.type() == CT_POINT || sat.type() == CT_EDGE)
        {
            Vec3f contactPoint = sat.pointB() - m.normal * radiusB;
            m.pushContact(contactPoint, depth);
        }
        else if (sat.face() == CT_TRIA)
        {
            Vec3f q[2]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * radiusB;
            int num = ClippingWithTri(q, sat.tri(), segB, transA, transB);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
        else if (sat.face() == CT_RECTA)
        {
            Vec3f q[2]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * radiusB;
            int num = ClippingWithRect(q, sat.rect(), segB, transA, transB);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
    };

    /*
    // return the closest point on the ray
    template<typename Real>
    DYN_FUNC inline Real rayInterRound2DHalfSeg(
		const Vec2f rayO,       //  Ray's Origin arbitraryon 2D Material Space
		const Vec2f rayD,       //  Ray's Direction arbitraryon 2D Material Space
		const Real halfl,	   
        const Real radius)      // 2D HalfSeg Space (-halfl, 0) - (halfl,0), Normal: (0,1)
    {
		if (REAL_LESS(rayO.y, 0.f) && REAL_LESS(rayD.y, 0.f)) return -1.f;
        Real t = -1.f, para = REAL_infinity; Vec2f p;
        // check body line 
        if (!REAL_EQUAL(rayD.y, 0.f))
        {
			t = (radius - rayO.y) / rayD.y;
            p = rayO + rayD * t;
            if (REAL_GREAT(t, 0.f) && REAL_LESS(t, para) && REAL_GREAT(p.x, - halfl) && REAL_LESS(p.x, +halfl)) para = t;
        }
        // check left caps
		Real a = rayD.x * rayD.x + rayD.y * rayD.y;
        Real b = 2 * ((rayO.x - halfl) * rayD.x + rayO.y * rayD.y);
		Real c = (rayO.x - halfl) * (rayO.x - halfl) + rayO.y * rayO.y - radius * radius;
		Real h = b * b - 4 * a * c;
		if (REAL_GREAT(h, 0.f))
		{
            Real sh = sqrt(h);
			t = (-b - sh) / (2 * a);
			p = rayO + rayD * t;
            if (REAL_GREAT(t, 0.f) && REAL_LESS(t, para) && REAL_GREAT(p.y, 0.f) && REAL_LESS(p.x, -halfl)) para = t;
            t = (-b + sh) / (2 * a);
            p = rayO + rayD * t;
            if (REAL_GREAT(t, 0.f) && REAL_LESS(t, para) && REAL_GREAT(p.y, 0.f) && REAL_LESS(p.x, -halfl)) para = t;
		}
        // check right caps
        b += 4 * halfl * rayD.x;
        c += halfl * halfl + 4 * halfl * rayD.x;
        h = b * b - 4 * a * c;
        if (REAL_GREAT(h, 0.f))
        {
            Real sh = sqrt(h);
            t = (-b - sh) / (2 * a);
            p = rayO + rayD * t;
            if (REAL_GREAT(t, 0.f) && REAL_LESS(t, para) && REAL_GREAT(p.y, 0.f) && REAL_GREAT(p.x, +halfl)) para = t;
            t = (-b + sh) / (2 * a);
            p = rayO + rayD * t;
            if (REAL_GREAT(t, 0.f) && REAL_LESS(t, para) && REAL_GREAT(p.y, 0.f) && REAL_GREAT(p.x, +halfl)) para = t;
        }

        return para;
    }

    template<typename Real>
    DYN_FUNC inline Vec3f rayInterRoundSeg(
        const Vec3f rayOrigin, // on seg
		const Vec3f rayDir, // along outside normal
        Segment3D seg,
        Real radius // radius > 0
        )
    {
        Vec3f dir = seg.direction(); dir.normalize();
        Vec3f d, a, f, c;
        Real t = dir.dot(rayDir), len2, sum;
        if (REAL_LESS(t, 0.f))
        {
            if (REAL_EQUAL(t, -1.f) || REAL_EQUAL((rayOrigin - seg.v0).normSquared(), 0.f)) return seg.v0 + rayDir * radius;
			d = seg.v0 - rayOrigin;
        }
        else 
        {
            if (REAL_EQUAL(t, 1.f) || REAL_EQUAL((rayOrigin - seg.v1).normSquared(), 0.f)) return seg.v1 + rayDir * radius;
            d = seg.v1 - rayOrigin;
        }
        

        f = d.cross(rayDir);

        a = f.cross(d);
        a.normalize();
        a *= radius;

        c = d + a;

		t = c.dot(a) / rayDir.dot(a);
        len2 = c.normSquared();
        if (t * t < len2)
        {
			return rayOrigin + rayDir * t;
        }

        len2 = d.normSquared();
        sum = (d[0] + d[1] + d[2]);
        t = -2 * sum + sqrt(sum * sum - len2 + radius * radius);

        return rayOrigin + rayDir * t;
    }
    */

    // ---------------------------------------- [   Tri   ] ----------------------------------------
    template<typename Real>
    DYN_FUNC inline void projectOnAxis(
        Real& lowerBoundary,
        Real& upperBoundary,
        const Vec3f axisNormal,
        Triangle3D tri,
        const Real radius
    )
    {
        Real t = tri.v[0].dot(axisNormal); lowerBoundary = upperBoundary = t;
        t = tri.v[1].dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);
        t = tri.v[2].dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

        lowerBoundary -= radius;
        upperBoundary += radius;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithTri(
        Vector<Real, 3>* q,
        const TTriangle3D<Real>& triA,
        const TTriangle3D<Real>& triB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        Triangle3D triTA = Triangle3D(triA.v[0] + transA, triA.v[1] + transA, triA.v[2] + transA);
        Triangle3D triTB = Triangle3D(triB.v[0] + transB, triB.v[1] + transB, triB.v[2] + transB);

        int num = 0;
        Vec3f oA = triTA.v[0];
        Vec3f nA = triTA.normal();
        Vec3f poly[3];

        num = cgeo::intrTriWithPlane(poly, oA, nA, triTB.v[0], triTB.v[1], triTB.v[2]);
        if (num > 0) num = cgeo::intrPolyWithTri(q, num, poly, triTA.v[0], triTA.v[1], triTA.v[2]);

        return num;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithRect(
        Vector<Real, 3>* q,
        const TRectangle3D<Real>& rectA,
        const TTriangle3D<Real>& triB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        Rectangle3D rectT = rectA; rectT.center += transA;
        Triangle3D triT = Triangle3D(triB.v[0] + transB, triB.v[1] + transB, triB.v[2] + transB);

        int num = 0;
        Vec3f oA = rectT.center;
        Vec3f nA = rectT.normal();
        Vec3f poly[3];

        num = cgeo::intrTriWithPlane(poly, oA, nA, triT.v[0], triT.v[1], triT.v[2]);
        if (num > 0) num = cgeo::intrPolyWithRect(q, num, poly, rectT.vertex(0).origin, rectT.vertex(1).origin, rectT.vertex(2).origin, rectT.vertex(3).origin);
        return num;
    }

    template<typename Real, typename Shape>
    DYN_FUNC inline void setupContactOnTri(
        TManifold<Real>& m,
        TSeparationData<Real>& sat,
        const Shape& shapeA,
        const TTriangle3D<Real>& triB,
        const Real radiusA,
        const Real radiusB)
    {
        Real depth = sat.depth();
        if (sat.type() == CT_POINT || sat.type() == CT_EDGE)
        {
            Vec3f contactPoint = sat.pointB();
            m.pushContact(contactPoint, depth);
        }
        else if (sat.face() == CT_TRIA)
        {
            Vec3f q[6]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * radiusB;
            int num = ClippingWithTri(q, sat.tri(), triB, transA, transB);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
        else if (sat.face() == CT_RECTA)
        {
            Vec3f q[8]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * radiusB;
            int num = ClippingWithRect(q, sat.rect(), triB, transA, transB);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
        else if (sat.face() == CT_TRIB)
        {
            Vec3f q[6]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * (radiusB);
            int num = ClippingWithTri(q, sat.tri(), shapeA, transB, transA);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
    };

    // ---------------------------------------- [   Tet   ] ----------------------------------------
    template<typename Real>
    DYN_FUNC inline void projectOnAxis(
        Real& lowerBoundary,
        Real& upperBoundary,
        const Vec3f axisNormal,
        Tet3D tet,
        const Real radius
    )
    {
        Real t = tet.v[0].dot(axisNormal); lowerBoundary = upperBoundary = t;
        t = tet.v[1].dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);
        t = tet.v[2].dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);
        t = tet.v[3].dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

        lowerBoundary -= radius;
        upperBoundary += radius;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithTri(
        Vector<Real, 3>* q,
        const TTriangle3D<Real>& triA,
        const TTet3D<Real>& tetB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        Triangle3D triTA = Triangle3D(triA.v[0] + transA, triA.v[1] + transA, triA.v[2] + transA);
        Tet3D tetTB = Tet3D(tetB.v[0] + transB, tetB.v[1] + transB, tetB.v[2] + transB, tetB.v[3] + transB);

        int num = 0;
        Vec3f oA = triTA.v[0];
        Vec3f nA = triTA.normal();
        Vec3f poly[4];

        num = cgeo::intrTetWithPlane(poly, oA, nA, tetTB.v[0], tetTB.v[1], tetTB.v[2], tetTB.v[3]);
        if (num > 0) num = cgeo::intrPolyWithTri(q, num, poly, triTA.v[0], triTA.v[1], triTA.v[2]);

        return num;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithRect(
        Vector<Real, 3>* q,
        const TRectangle3D<Real>& rectA,
        const TTet3D<Real>& tetB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        Rectangle3D rectT = rectA; rectT.center += transA;
        Tet3D tetT = Tet3D(tetB.v[0] + transB, tetB.v[1] + transB, tetB.v[2] + transB, tetB.v[3] + transB);

        int num = 0;
        Vec3f oA = rectT.center;
        Vec3f nA = rectT.normal();
        Vec3f poly[4];

        num = cgeo::intrTetWithPlane(poly, oA, nA, tetT.v[0], tetT.v[1], tetT.v[2], tetT.v[3]);
        if (num > 0) num = cgeo::intrPolyWithRect(q, num, poly, rectT.vertex(0).origin, rectT.vertex(1).origin, rectT.vertex(2).origin, rectT.vertex(3).origin);
        return num;
    }

    template<typename Real, typename Shape>
    DYN_FUNC inline void setupContactOnTet(
        TManifold<Real>& m,
        TSeparationData<Real>& sat,
        const Shape& shapeA,
        const TTet3D<Real>& tetB,
        const Real radiusA,
        const Real radiusB)
    {
        Real depth = sat.depth();
        if (sat.type() == CT_POINT || sat.type() == CT_EDGE)
        {
            Vec3f contactPoint = sat.pointB();
            m.pushContact(contactPoint, depth);
        }
        else if (sat.face() == CT_TRIA)
        {
            Vec3f q[6]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * radiusB;
            int num = ClippingWithTri(q, sat.tri(), tetB, transA, transB);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
        else if (sat.face() == CT_RECTA)
        {
            Vec3f q[8]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * radiusB;
            int num = ClippingWithRect(q, sat.rect(), tetB, transA, transB);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
        else if (sat.face() == CT_TRIB)
        {
            Vec3f q[6]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * (radiusB);
            int num = ClippingWithTri(q, sat.tri(), shapeA, transB, transA);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
    };

    // ---------------------------------------- [   Box   ] ----------------------------------------
    template<typename Real>
    DYN_FUNC inline void projectOnAxis(
        Real& lowerBoundary,
        Real& upperBoundary,
        const Vec3f axisNormal,
        OrientedBox3D box,
        const Real radius
    )
    {
        Vec3f center = box.center;
        Vec3f u = box.u;
        Vec3f v = box.v;
        Vec3f w = box.w;
        Vec3f extent = box.extent;
        Vec3f p;

        p = (center - u * extent[0] - v * extent[1] - w * extent[2]);
        Real t = p.dot(axisNormal); lowerBoundary = upperBoundary = t;

        p = (center - u * extent[0] - v * extent[1] + w * extent[2]);
        t = p.dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

        p = (center - u * extent[0] + v * extent[1] - w * extent[2]);
        t = p.dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

		p = (center - u * extent[0] + v * extent[1] + w * extent[2]);
		t = p.dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

		p = (center + u * extent[0] - v * extent[1] - w * extent[2]);
		t = p.dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

		p = (center + u * extent[0] - v * extent[1] + w * extent[2]);
		t = p.dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

		p = (center + u * extent[0] + v * extent[1] - w * extent[2]);
		t = p.dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

		p = (center + u * extent[0] + v * extent[1] + w * extent[2]);
		t = p.dot(axisNormal); lowerBoundary = glm::min(lowerBoundary, t); upperBoundary = glm::max(upperBoundary, t);

        lowerBoundary -= radius;
        upperBoundary += radius;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithTri(
        Vector<Real, 3>* q,
        const TTriangle3D<Real>& triA,
        const TOrientedBox3D<Real>& boxB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        Triangle3D triTA = Triangle3D(triA.v[0] + transA, triA.v[1] + transA, triA.v[2] + transA);
        OrientedBox3D boxTB = boxB; boxTB.center += transB;

        int num = 0;
        Vec3f oA = triTA.v[0];
        Vec3f nA = triTA.normal();
        Vec3f poly[4];

        num = cgeo::intrBoxWithPlane(poly, oA, nA, boxTB.center, boxTB.u * boxTB.extent[0], boxTB.v * boxTB.extent[1], boxTB.w * boxTB.extent[2]);
        if (num > 0) num = cgeo::intrPolyWithTri(q, num, poly, triTA.v[0], triTA.v[1], triTA.v[2]);

        return num;
    }

    template<typename Real>
    DYN_FUNC inline int ClippingWithRect(
        Vector<Real, 3>* q,
        const TRectangle3D<Real>& rectA,
        const TOrientedBox3D<Real>& boxB,
        const Vector<Real, 3>& transA,
        const Vector<Real, 3>& transB
    )
    {
        Rectangle3D rectT = rectA; rectT.center += transA;
        OrientedBox3D boxTB = boxB; boxTB.center += transB;

        int num = 0;
        Vec3f oA = rectT.center;
        Vec3f nA = rectT.normal();
        Vec3f poly[4];

        num = cgeo::intrBoxWithPlane(poly, oA, nA, boxTB.center, boxTB.u * boxTB.extent[0], boxTB.v * boxTB.extent[1], boxTB.w * boxTB.extent[2]);
        if (num > 0) num = cgeo::intrPolyWithRect(q, num, poly, rectT.vertex(0).origin, rectT.vertex(1).origin, rectT.vertex(2).origin, rectT.vertex(3).origin);
        return num;
    }


    template<typename Real, typename Shape>
    DYN_FUNC inline void setupContactOnBox(
        TManifold<Real>& m,
        TSeparationData<Real>& sat,
        const Shape& shapeA,
        const TOrientedBox3D<Real>& boxB,
        const Real radiusA,
        const Real radiusB)
    {
        Real depth = sat.depth();
        if (sat.type() == CT_POINT || sat.type() == CT_EDGE)
        {
            Vec3f contactPoint = sat.pointB();
            m.pushContact(contactPoint, depth);
        }
        else if (sat.face() == CT_TRIA)
        {
            Vec3f q[6]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * radiusB;
            int num = ClippingWithTri(q, sat.tri(), boxB, transA, transB);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
        else if (sat.face() == CT_RECTA)
        {
            Vec3f q[8]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * radiusB;
            int num = ClippingWithRect(q, sat.rect(), boxB, transA, transB);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
        else if (sat.face() == CT_RECTB)
        {
            Vec3f q[6]; Vec3f transA, transB;
            transA = m.normal * (radiusA + depth); // depth < 0
            transB = -m.normal * (radiusB);
            int num = ClippingWithRect(q, sat.rect(), shapeA, transB, transA);
            for (int i = 0; i < num; ++i) m.pushContact(q[i], depth);
        }
    };

    // ---------------------------------------- [Sphere - Sphere] ----------------------------------------

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Sphere3D& sphereA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB) 
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, sphereA, sphereB, radiusA, radiusB, pA, pB, sphereA.radius, sphereB.radius); };
        
        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's Point to A's Point
        checkAxisP(sphereA.center, sphereB.center);
    }



    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphereA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereB
        MSDF(sat, sphereA, sphereB, radiusA, radiusB);

        if (REAL_LESS(sat.depth(), 0) && REAL_GREAT(sat.depth(), -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on sphereB

            setupContactOnSphere(m, sat, sphereB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    // ---------------------------------------- [Seg - Sphere] ----------------------------------------

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Segment3D& segA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, segA, sphereB, radiusA, radiusB, pA, pB, 0.f, sphereB.radius); };

        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's Point to A's Point
        Vec3f pB = sphereB.center;
        Point3D queryP(pB);
        Point3D projP = queryP.project(segA);
        checkAxisP(projP.origin, pB);
    }


    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphereA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereA
        MSDF(sat, segB, sphereA, radiusB, radiusA);
		
        Real depth = sat.depth();
        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            sat.reverse();              // contact normal on segB
            m.normal = sat.normal(); 
            /*
            auto setupContactOnSeg = [&]()
            {
				Vec3f contactPoint = sat.pointB() - m.normal * (radiusB);
                m.pushContact(contactPoint, depth);
            };
            */

            setupContactOnSeg(m, sat, segB, radiusA, radiusB);

        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Segment3D& segA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereB
        MSDF(sat, segA, sphereB, radiusA, radiusB);
        Real depth = sat.depth();
        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on sphereB

            /*
            auto setupContactOnSphere = [&]()
            {
                Vec3f contactPoint = sat.pointB()- m.normal * (sphereB.radius);
                m.pushContact(contactPoint, depth);
            };
            */

            setupContactOnSphere(m, sat, sphereB, radiusA, radiusB);

        }
        else m.contactCount = 0;
    }

    // ---------------------------------------- [Tri - Sphere] ----------------------------------------
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Triangle3D& triA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, triA, sphereB, radiusA, radiusB, pA, pB, 0.f, sphereB.radius); };

        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's Point to A's Point
        Vec3f pB = sphereB.center;
        Point3D queryP(pB);
        Point3D projP = queryP.project(triA);
        checkAxisP(projP.origin, pB);
    }



    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphereA, const Triangle3D& triB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereA
        MSDF(sat, triB, sphereA, radiusB, radiusA);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
			sat.reverse();              // contact normal on triB
            m.normal = sat.normal(); 

            setupContactOnTri(m, sat, sphereA, triB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Triangle3D& triA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereB
        MSDF(sat, triA, sphereB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on sphereB

            /*
            auto setupContactOnSphere = [&]()
            {
                Vec3f contactPoint = sat.pointB() - m.normal * (sphereB.radius);
                m.pushContact(contactPoint, depth);
            };
            */

            setupContactOnSphere(m, sat, sphereB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    // ---------------------------------------- [Tet - Sphere] ----------------------------------------
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Tet3D& tetA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, tetA, sphereB, radiusA, radiusB, pA, pB, 0.f, sphereB.radius); };

        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's Point to A's Point
        Vec3f pB = sphereB.center;
        Point3D queryP(pB);
        Point3D projP = queryP.project(tetA);
        checkAxisP(projP.origin, pB);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphereA, const Tet3D& tetB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereA
        MSDF(sat, tetB, sphereA, radiusB, radiusA);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            sat.reverse();              // contact normal on tetB
            m.normal = sat.normal();

            setupContactOnTet(m, sat, sphereA, tetB, radiusA, radiusB);

        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tetA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereB
        MSDF(sat, tetA, sphereB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on sphereB

            /*
            auto setupContactOnSphere = [&]()
            {
                Vec3f contactPoint = sat.pointB() - m.normal * (sphereB.radius);
                m.pushContact(contactPoint, depth);
            };
            */

            setupContactOnSphere(m, sat, sphereB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    // ---------------------------------------- [Box - Sphere] ----------------------------------------
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const OBox3D& boxA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, boxA, sphereB, radiusA, radiusB, pA, pB, 0.f, sphereB.radius); };

        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's Point to A's Point
        Vec3f pB = sphereB.center;
        Point3D queryP(pB);
        Point3D projP = queryP.project(boxA);
        checkAxisP(projP.origin, pB);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphereA, const OBox3D& boxB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereA
        MSDF(sat, boxB, sphereA, radiusB, radiusA);
        Real depth = sat.depth();
        //printf("Dep :%f\n", depth);
        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            sat.reverse();              // contact normal on boxB
            m.normal = sat.normal();

            setupContactOnBox(m, sat, sphereA, boxB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& boxA, const Sphere3D& sphereB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on sphereB
        MSDF(sat, boxA, sphereB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on sphereB

            setupContactOnSphere(m, sat, sphereB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }


    // ---------------------------------------- [Seg - Seg] ----------------------------------------
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Segment3D& segA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, segA, segB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, segA, segB, radiusA, radiusB, edgeA, edgeB);};

        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 2; j++)
        {
            Vec3f pB = (j == 0) ? (segB.v0) : (segB.v1);
            Point3D queryP(pB);
            Point3D projP = queryP.project(segA);
            checkAxisP(projP.origin, pB);
        }
        for (int j = 0; j < 2; j++)
        {
            Vec3f pA = (j == 0) ? (segA.v0) : (segA.v1);
            Point3D queryP(pA);
            Point3D projP = queryP.project(segB);
            checkAxisP(pA, projP.origin);
        }

        // Minkowski Edge-Edge Normal
        // segA x segB
        checkAxisE(segA, segB);
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const Segment3D& segA, const Segment3D& segB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, segA, segB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 2; j++)
        {
            Vec3f pB = (j == 0) ? (segB.v0) : (segB.v1);
            Point3D queryP(pB);
            Point3D projP = queryP.project(segA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }

        // Minkowski Face Normal
        // 1. segA x segB
        Vec3f dirSegA = segA.direction();
        Vec3f dirSegB = segB.direction();
        axisTmp = dirSegA.cross(dirSegB);
        checkAxis(axisTmp);
    }
    */

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Segment3D& segA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on segB
        MSDF(sat, segA, segB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on segB
            /*
            auto setupContactOnSeg = [&]()
			{
                Vec3f contactPoint = sat.pointB() - m.normal * radiusB;
				m.pushContact(contactPoint, depth);
			};
            */

            setupContactOnSeg(m, sat, segB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    // ---------------------------------------- [Tri - Seg] ----------------------------------------

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Triangle3D& triA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, triA, segB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, triA, segB, radiusA, radiusB, edgeA, edgeB); };
        auto checkAxisT = [&](Triangle3D face) { checkAxisTri(sat, triA, segB, radiusA, radiusB, face, SeparationType::CT_TRIA); };

        // Minkowski Face Normal
        // tri face
        checkAxisT(triA);
        
        // Minkowski Edge-Edge Normal
        // segA x segB
        checkAxisE(Segment3D(triA.v[0], triA.v[1]), segB);
        checkAxisE(Segment3D(triA.v[0], triA.v[2]), segB);
        checkAxisE(Segment3D(triA.v[1], triA.v[2]), segB);


        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 2; j++)
        {
            Vec3f pB = (j == 0) ? (segB.v0) : (segB.v1);
            Point3D queryP(pB);
            Point3D projP = queryP.project(triA);
            checkAxisP(projP.origin, pB);
        }

        for (int j = 0; j < 3; j++)
        {
            Vec3f pA = triA.v[j];
            Point3D queryP(pA);
            Point3D projP = queryP.project(segB);
            checkAxisP(pA, projP.origin);
        }



    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const Triangle3D& triA, const Segment3D& segB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, triA, segB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. tri face 
        axisTmp = triA.normal();
        checkAxis(axisTmp);

        // 2. edge cross
        Vec3f dirSeg = segB.direction();
        for (int i = 0; i < 2; i++)
            for (int j = i + 1; j < 3; ++j)
            {
                Vec3f triDir = triA.v[j] - triA.v[i];
                axisTmp = dirSeg.cross(triDir);
                checkAxis(axisTmp);
            }

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 2; j++)
        {
            Vec3f pB = (j == 0) ? (segB.v0) : (segB.v1);
            Point3D queryP(pB);
            Point3D projP = queryP.project(triA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }
    */

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Triangle3D& triA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on segB
        MSDF(sat, triA, segB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on segB


            setupContactOnSeg(m, sat, segB, radiusA, radiusB);

        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Segment3D& segA, const Triangle3D& triB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on segA
        MSDF(sat, triB, segA, radiusB, radiusA);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
			sat.reverse();              // contact normal on triB
            m.normal = sat.normal(); // contact normal on triB

            setupContactOnTri(m, sat, segA, triB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }


    // ---------------------------------------- [Tet - Seg] ----------------------------------------
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Tet3D& tetA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, tetA, segB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, tetA, segB, radiusA, radiusB, edgeA, edgeB); };
        auto checkAxisT = [&](Triangle3D face) { checkAxisTri(sat, tetA, segB, radiusA, radiusB, face, SeparationType::CT_TRIA); };
        
        // Minkowski Face Normal
        // tet face
        checkAxisT(tetA.face(0));
        checkAxisT(tetA.face(1));
        checkAxisT(tetA.face(2));
        checkAxisT(tetA.face(3));
        
        // Minkowski Edge-Edge Normal
        // tetA x segB
        checkAxisE(Segment3D(tetA.v[0], tetA.v[1]), segB);
        checkAxisE(Segment3D(tetA.v[0], tetA.v[2]), segB);
        checkAxisE(Segment3D(tetA.v[1], tetA.v[2]), segB);
        checkAxisE(Segment3D(tetA.v[0], tetA.v[3]), segB);
        checkAxisE(Segment3D(tetA.v[1], tetA.v[3]), segB);
        checkAxisE(Segment3D(tetA.v[2], tetA.v[3]), segB);


        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 2; j++)
        {
            Vec3f pB = (j == 0) ? (segB.v0) : (segB.v1);
            Point3D queryP(pB);
            Point3D projP = queryP.project(tetA);
            checkAxisP(projP.origin, pB);
        }

        for (int j = 0; j < 4; j++)
        {
            Vec3f pA = tetA.v[j];
            Point3D queryP(pA);
            Point3D projP = queryP.project(segB);
            checkAxisP(pA, projP.origin);
        }
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const Tet3D& tetA, const Segment3D& segB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, tetA, segB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. tet face (u, v, w)
        for (int i = 0; i < 4; ++i)
        {
            axisTmp = tetA.face(i).normal();
            checkAxis(axisTmp);
        }

        // 2. edge cross
        Vec3f dirSeg = segB.direction();
        for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 4; ++j)
            {
                Vec3f tetDir = tetA.v[j] - tetA.v[i];
                axisTmp = dirSeg.cross(tetDir);
                checkAxis(axisTmp);
            }

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 2; j++)
        {
            Vec3f pB = (j == 0) ? (segB.v0) : (segB.v1);
            Point3D queryP(pB);
            Point3D projP = queryP.project(tetA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }

    template<typename Real>
    DYN_FUNC inline void setupContactOnTet(
        Segment3D seg,
        Tet3D tet,
        const Real radiusA,
        const Real radiusB,
        const Real depth, // <0 
        TManifold<Real>& m)
    {
        //Gen contact point on tet

        int count = 0;
        Vec3f axisNormal = m.normal;

        auto setContact = [&](Segment3D minPQ, Real gap)
        {
            if (minPQ.length() < gap)
            {
                if (count >= 8) return;
                m.contacts[count].penetration = depth;
                m.contacts[count].position = minPQ.endPoint();
                count++;
            }
        };

        auto checkFace = [&](Vec3f v0, Vec3f v1, Vec3f v2, Vec3f Nz)
        {
            if (count >= 8) return;
            Triangle3D tri(v0, v1, v2);
            // inter & prox
            auto minPQ = seg.proximity(tri);
            // -: outside
            // +: inside 
            Real gap = (Nz.dot(minPQ.direction()) < 0) ? radiusA + radiusB : radiusB;
            setContact(minPQ, gap);

            // if parallel need to check the two points on segment
            for (int i = 0; i < 2; i++)
            {
                Vec3f v = (i == 0) ? seg.v0 : seg.v1;
                TPoint3D<Real> q = TPoint3D<Real>(v).project(tri);
                m.pushContact(q, depth);
            }
        };

        // Check intersection and proximity for each tri face
        Vec3f centerTet = tet.barycenter().origin;
        for (int i = 0; i < 4; i++)
        {
            // Outer norm on tri face.
            Vec3f outerNorm = tet.face(i).normal();
            Vec3f dir = centerTet - tet.face(i).v[0];
            if (dir.dot(outerNorm) > 0.f) outerNorm = -outerNorm;

            checkFace(tet.face(i).v[0], tet.face(i).v[1], tet.face(i).v[2], outerNorm);
        }

        m.contactCount = count;

        printf("tet contact:%d\n", count);
    }
    */

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Segment3D& segA, const Tet3D& tetB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on segA
        MSDF(sat, tetB, segA, radiusB, radiusA);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            sat.reverse();
            m.normal = sat.normal(); // contact normal on tetB

            setupContactOnTet(m, sat, segA, tetB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tetA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on segB
        MSDF(sat, tetA, segB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on segB

            setupContactOnSeg(m, sat, segB, radiusA, radiusB);

        }
        else m.contactCount = 0;
    }


    // ---------------------------------------- [Box - Seg] ----------------------------------------

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const OBox3D& boxA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, boxA, segB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, boxA, segB, radiusA, radiusB, edgeA, edgeB); };
        //auto checkAxisT = [&](Triangle3D face) { checkAxisTri(sat, boxA, segB, radiusA, radiusB, face, SeparationType::CT_TRIA); };
        auto checkAxisR = [&](Rectangle3D face) { checkAxisRect(sat, boxA, segB, radiusA, radiusB, face, SeparationType::CT_RECTA); };

        // Minkowski Face Normal
        // box face
        for (int j = 0; j < 6; j++)
        {
            Rectangle3D faceA = boxA.face(j);
            checkAxisR(faceA);
        }

        // Minkowski Edge-Edge Normal
        // boxA x segB
        for (int j = 0; j < 12; j++)
        {
            Segment3D edgeA = boxA.edge(j);
            checkAxisE(edgeA, segB);
        }

        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 2; j++)
        {
            Vec3f pB = (j == 0) ? (segB.v0) : (segB.v1);
            Point3D queryP(pB);
            Point3D projP = queryP.project(boxA);
            checkAxisP(projP.origin, pB);
        }

        for (int j = 0; j < 8; j++)
        {
            Vec3f pA = boxA.vertex(j).origin;
            Point3D queryP(pA);
            Point3D projP = queryP.project(segB);
            checkAxisP(pA, projP.origin);
        }
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const OBox3D& boxA, const Segment3D& segB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, boxA, segB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. box face (u, v, w)
        axisTmp = boxA.u / boxA.u.norm(); checkAxis(axisTmp);
        axisTmp = boxA.v / boxA.v.norm(); checkAxis(axisTmp);
        axisTmp = boxA.w / boxA.w.norm(); checkAxis(axisTmp);
        // 2. edge cross
        Vec3f dirSeg = segB.direction();
        for (int j = 0; j < 3; j++)
        {
            Vec3f boxDir = (j == 0) ? (boxA.u) : ((j == 1) ? (boxA.v) : (boxA.w));
            axisTmp = dirSeg.cross(boxDir);
            checkAxis(axisTmp);
        }

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 2; j++)
        {
            Vec3f pB = (j == 0) ? (segB.v0) : (segB.v1);
            Point3D queryP(pB);
            Point3D projP = queryP.project(boxA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }
    

    template<typename Real>
    DYN_FUNC inline void setupContactOnBox(
        Segment3D seg,
        OrientedBox3D box,
        const Real radiusA,
        const Real radiusB,
        const Real depth, // <0 
        TManifold<Real>& m)
    {
        //Gen contact point on box

        // Intersecte on Margin     : F-V, V-F, E-E

        // Intersecte on Polygon    : E-F, F-E, B-V


        int count = 0;
        Vec3f axisNormal = m.normal;

        auto setContact = [&](Segment3D minPQ, Real gap)
        {
            if (minPQ.length() < gap)
            {
                if (count >= 8) return;
                m.contacts[count].penetration = depth;
                m.contacts[count].position = minPQ.endPoint();
                count++;
            }
        };

        auto checkFace = [&](Vec3f c, Vec3f Nx, Vec3f Ny, Vec3f Nz, Real Ex, Real Ey)
        {
            if (count >= 8) return;
            Rectangle3D rect(c, Nx, Ny, Vec2f(Ex, Ey));
            // inter & prox
            auto minPQ = seg.proximity(rect);
            Real gap = (Nz.dot(minPQ.direction()) < 0) ? radiusA + radiusB : radiusA;
            setContact(minPQ, gap);

            // if parallel need to check the two points on segment
            for (int i = 0; i < 2; i++)
            {
                Vec3f v = (i == 0) ? seg.v0 : seg.v1;
                TPoint3D<Real> p(v);
                TPoint3D<Real> q = TPoint3D<Real>(p).project(rect);
                TSegment3D<Real> pq = q - p;
                setContact(pq, gap);
            }
        };

        // Check intersection and proximity for each face
        // -u +u
        for (int iu = -1; iu <= 1; iu += 2)
        {
            Vec3f c = box.center + box.u * box.extent[0] * iu;
            checkFace(c, box.v, box.w, box.u * Real(iu), box.extent[1], box.extent[2]);
        }
        // -v +v
        for (int iv = -1; iv <= 1; iv += 2)
        {
            Vec3f c = box.center + box.v * box.extent[1] * iv;
            checkFace(c, box.u, box.w, box.v * Real(iv), box.extent[0], box.extent[2]);
        }
        // -w +w
        for (int iw = -1; iw <= 1; iw += 2)
        {
            Vec3f c = box.center + box.w * box.extent[2] * iw;
            checkFace(c, box.u, box.v, box.w * Real(iw), box.extent[0], box.extent[1]);
        }

        m.contactCount = count;

        printf("box contact:%d\n", count);
    }
    */


    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Segment3D& segA, const OBox3D& boxB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on segA
        MSDF(sat, boxB, segA, radiusB, radiusA);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            sat.reverse();
            m.normal = sat.normal(); // contact normal on boxB

            setupContactOnBox(m, sat, segA, boxB, radiusA, radiusB);

        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& boxA, const Segment3D& segB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on segB
        MSDF(sat, boxA, segB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on segB

            setupContactOnSeg(m, sat, segB, radiusA, radiusB);

        }
        else m.contactCount = 0;
    }


    // ---------------------------------------- [Tri - Tri] ----------------------------------------

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Triangle3D& triA, const Triangle3D& triB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, triA, triB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, triA, triB, radiusA, radiusB, edgeA, edgeB); };
        auto checkAxisT = [&](Triangle3D face, auto type) { checkAxisTri(sat, triA, triB, radiusA, radiusB, face, type); };
        
        // Minkowski Face Normal
        // tri face
        checkAxisT(triA, SeparationType::CT_TRIA);
        checkAxisT(triB, SeparationType::CT_TRIB);


        // Minkowski Edge-Edge Normal
        // triA x triB
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
                int ni = (i == 2) ? 0 : i + 1;
                int nj = (j == 2) ? 0 : j + 1;
				checkAxisE(Segment3D(triA.v[i], triA.v[ni]), Segment3D(triB.v[j], triB.v[nj]));
			}

        // Minkowski Point-Point Normal
        for (int j = 0; j < 3; j++)
        {
            Vec3f pA = triA.v[j];
            Point3D queryP(pA);
            Point3D projP = queryP.project(triB);
            checkAxisP(pA, projP.origin);
        }

        for (int j = 0; j < 3; j++)
        {
            Vec3f pB = triB.v[j];
            Point3D queryP(pB);
            Point3D projP = queryP.project(triA);
            checkAxisP(projP.origin, pB);
        }
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const Triangle3D& triA, const Triangle3D& triB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, triA, triB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. triA face
        axisTmp = triA.normal();
        checkAxis(axisTmp);

        // 2. triB face
        axisTmp = triB.normal();
        checkAxis(axisTmp);

        // 3. triA's edge x triB's edge
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                Vec3f triDirA = triA.v[(i + 1) % 3] - triA.v[i];
                Vec3f triDirB = triB.v[(j + 1) % 3] - triB.v[j];
                axisTmp = triDirA.cross(triDirB);
                checkAxis(axisTmp);
            }

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 3; j++)
        {
            Vec3f pB = triB.v[j];
            Point3D queryP(pB);
            Point3D projP = queryP.project(triA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }
    */

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Triangle3D& triA, const Triangle3D& triB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on triB
        MSDF(sat, triA, triB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on triB
            setupContactOnTri(m, sat, triA, triB, radiusA, radiusB);

        }
        else m.contactCount = 0;
    }


    // ---------------------------------------- [Tet - Tri] ----------------------------------------

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Tet3D& tetA, const Triangle3D& triB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, tetA, triB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, tetA, triB, radiusA, radiusB, edgeA, edgeB); };
        auto checkAxisT = [&](Triangle3D face, auto type) { checkAxisTri(sat, tetA, triB, radiusA, radiusB, face, type); };

        // Minkowski Face Normal
        // face
        checkAxisT(tetA.face(0), SeparationType::CT_TRIA);
        checkAxisT(tetA.face(1), SeparationType::CT_TRIA);
        checkAxisT(tetA.face(2), SeparationType::CT_TRIA);
        checkAxisT(tetA.face(3), SeparationType::CT_TRIA);
        checkAxisT(triB, SeparationType::CT_TRIB);


        // Minkowski Edge-Edge Normal
        // tetA x triB
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 3; j++)
            {
                int nj = (j == 2) ? 0 : j + 1;
                checkAxisE(tetA.edge(i), Segment3D(triB.v[j], triB.v[nj]));
            }

        // Minkowski Point-Point Normal
        for (int j = 0; j < 4; j++)
        {
            Vec3f pA = tetA.v[j];
            Point3D queryP(pA);
            Point3D projP = queryP.project(triB);
            checkAxisP(pA, projP.origin);
        }

        for (int j = 0; j < 3; j++)
        {
            Vec3f pB = triB.v[j];
            Point3D queryP(pB);
            Point3D projP = queryP.project(tetA);
            checkAxisP(projP.origin, pB);
        }
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const Tet3D& tetA, const Triangle3D& triB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, tetA, triB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. tet face (u, v, w)
        for (int i = 0; i < 4; ++i)
        {
            axisTmp = tetA.face(i).normal();
            checkAxis(axisTmp);
        }

        // 2. Tri face
        axisTmp = triB.normal();
        checkAxis(axisTmp);

        // 2. edge cross
        for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 4; ++j)
            {
                Vec3f tetDir = tetA.v[j] - tetA.v[i];
                for (int k = 0; k < 3; k++)
                {
                    Vec3f triDir = triB.v[(k + 1) % 3] - triB.v[k];
                    axisTmp = tetDir.cross(triDir);
                    checkAxis(axisTmp);
                }
            }

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 3; j++)
        {
            Vec3f pB = triB.v[j];
            Point3D queryP(pB);
            Point3D projP = queryP.project(tetA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }
    */

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Triangle3D& triA, const Tet3D& tetB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on triA
        MSDF(sat, tetB, triA, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            sat.reverse();          // contact normal on tetB
            m.normal = sat.normal(); 

            setupContactOnTet(m, sat, triA, tetB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tetA, const Triangle3D& triB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on triB
        MSDF(sat, tetA, triB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on triB

            setupContactOnTri(m, sat, tetA, triB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    // ---------------------------------------- [Box - Tri] ----------------------------------------

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const OBox3D& boxA, const Triangle3D& triB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, boxA, triB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, boxA, triB, radiusA, radiusB, edgeA, edgeB); };
        auto checkAxisT = [&](Triangle3D face) { checkAxisTri(sat, boxA, triB, radiusA, radiusB, face, CT_TRIB); };
        auto checkAxisR = [&](Rectangle3D face) { checkAxisRect(sat, boxA, triB, radiusA, radiusB, face, CT_RECTA); };

        // Minkowski Face Normal
        // face
        for (int j = 0; j < 6; j++) checkAxisR(boxA.face(j));
        checkAxisT(triB);

        // Minkowski Edge-Edge Normal
        // boxA x triB
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 3; j++)
            {
                int nj = (j == 2) ? 0 : j + 1;
                checkAxisE(boxA.edge(i), Segment3D(triB.v[j], triB.v[nj]));
            }

        // Minkowski Point-Point Normal
        for (int j = 0; j < 8; j++)
        {
            Vec3f pA = boxA.vertex(j).origin;
            Point3D queryP(pA);
            Point3D projP = queryP.project(triB);
            checkAxisP(pA, projP.origin);
        }

        for (int j = 0; j < 3; j++)
        {
            Vec3f pB = triB.v[j];
            Point3D queryP(pB);
            Point3D projP = queryP.project(boxA);
            checkAxisP(projP.origin, pB);
        }
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const OBox3D& boxA, const Triangle3D& triB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, boxA, triB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. Box face (u, v, w)
        axisTmp = boxA.u / boxA.u.norm(); checkAxis(axisTmp);
        axisTmp = boxA.v / boxA.v.norm(); checkAxis(axisTmp);
        axisTmp = boxA.w / boxA.w.norm(); checkAxis(axisTmp);

        // 2. Tri face
        axisTmp = triB.normal();
        checkAxis(axisTmp);

        // 2. edge cross
        for (int i = 0; i < 3; i++)
        {
            Vec3f boxDir = (i == 0) ? (boxA.u) : ((i == 1) ? (boxA.v) : (boxA.w));
            for (int j = 0; j < 3; j++)
            {
                Vec3f triDir = triB.v[(j + 1) % 3] - triB.v[j];
                axisTmp = boxDir.cross(triDir);
                checkAxis(axisTmp);
            }
        }
        
        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 3; j++)
        {
            Vec3f pB = triB.v[j];
            Point3D queryP(pB);
            Point3D projP = queryP.project(boxA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }
    */


    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Triangle3D& triA, const OBox3D& boxB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on triA
        MSDF(sat, boxB, triA, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            sat.reverse();          // contact normal on boxB
            m.normal = sat.normal();

            setupContactOnBox(m, sat, triA, boxB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& boxA, const Triangle3D& triB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on triB
        MSDF(sat, boxA, triB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on triB

            setupContactOnTri(m, sat, boxA, triB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }


    // ---------------------------------------- [Tet - Tet] ----------------------------------------
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const Tet3D& tetA, const Tet3D& tetB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, tetA, tetB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, tetA, tetB, radiusA, radiusB, edgeA, edgeB); };
        auto checkAxisT = [&](Triangle3D face, auto type) { checkAxisTri(sat, tetA, tetB, radiusA, radiusB, face, type); };

        // Minkowski Face Normal
        // tet face
        checkAxisT(tetA.face(0), SeparationType::CT_TRIA);
        checkAxisT(tetA.face(1), SeparationType::CT_TRIA);
        checkAxisT(tetA.face(2), SeparationType::CT_TRIA);
        checkAxisT(tetA.face(3), SeparationType::CT_TRIA);

        checkAxisT(tetB.face(0), SeparationType::CT_TRIB);
        checkAxisT(tetB.face(1), SeparationType::CT_TRIB);
        checkAxisT(tetB.face(2), SeparationType::CT_TRIB);
        checkAxisT(tetB.face(3), SeparationType::CT_TRIB);

        // Minkowski Edge-Edge Normal
        // tetA x tetB
		for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
            {
				Segment3D edgeA = tetA.edge(i);
				Segment3D edgeB = tetB.edge(j);
				checkAxisE(edgeA, edgeB);
			}

        // Minkowski Point-Point Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 4; j++)
        {
            Vec3f pA = tetB.v[j];
            Point3D queryP(pA);
            Point3D projP = queryP.project(tetA);
            checkAxisP(pA, projP.origin);
        }

        for (int j = 0; j < 4; j++)
        {
            Vec3f pA = tetA.v[j];
            Point3D queryP(pA);
            Point3D projP = queryP.project(tetB);
            checkAxisP(pA, projP.origin);
        }
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const Tet3D& tetA, const Tet3D& tetB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, tetA, tetB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. tetA face
        for (int i = 0; i < 4; i++)
        {
            axisTmp = tetA.face(i).normal();
            checkAxis(axisTmp);
        }

        // 2. tetB face
        for (int i = 0; i < 4; i++)
        {
            axisTmp = tetB.face(i).normal();
            checkAxis(axisTmp);
        }

        // 3. tetA's edge x tetB's edge
        for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 4; ++j)
            {
                Vec3f tetDirA = tetA.v[j] - tetA.v[i];
                for (int k = 0; k < 3; k++)
                    for (int l = k + 1; l < 4; l++)
                    {
                        Vec3f tetDirB = tetB.v[k] - tetB.v[l];
                        axisTmp = tetDirA.cross(tetDirB);
                        checkAxis(axisTmp);
                    }
            }

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int i = 0; i <4; i++)
        {
            Vec3f pB = tetB.v[i];
            Point3D queryP(pB);
            Point3D projP = queryP.project(tetA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }
    */

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tetA, const Tet3D& tetB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on tetB
        MSDF(sat, tetA, tetB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on tetB

            setupContactOnTet(m, sat, tetA, tetB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    // ---------------------------------------- [Box - Tet] ----------------------------------------

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const OBox3D& boxA, const Tet3D& tetB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, boxA, tetB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, boxA, tetB, radiusA, radiusB, edgeA, edgeB); };
        auto checkAxisT = [&](Triangle3D face) { checkAxisTri(sat, boxA, tetB, radiusA, radiusB, face, CT_TRIB); };
        auto checkAxisR = [&](Rectangle3D face) { checkAxisRect(sat, boxA, tetB, radiusA, radiusB, face, CT_RECTA); };

        // Minkowski Face Normal
        // face
        // box face
        for (int j = 0; j < 6; j++) checkAxisR(boxA.face(j));
        checkAxisT(tetB.face(0));
        checkAxisT(tetB.face(1));
        checkAxisT(tetB.face(2));
        checkAxisT(tetB.face(3));


        // Minkowski Edge-Edge Normal
        // boxA x tetB
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 6; j++)
            {
                checkAxisE(boxA.edge(i), tetB.edge(j));
            }

        // Minkowski Point-Point Normal
        for (int j = 0; j < 8; j++)
        {
            Vec3f pA = boxA.vertex(j).origin;
            Point3D queryP(pA);
            Point3D projP = queryP.project(tetB);
            checkAxisP(pA, projP.origin);
        }

        for (int j = 0; j < 4; j++)
        {
            Vec3f pB = tetB.v[j];
            Point3D queryP(pB);
            Point3D projP = queryP.project(boxA);
            checkAxisP(projP.origin, pB);
        }
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const OBox3D& boxA, const Tet3D& tetB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, boxA, tetB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. Box face (u, v, w)
        axisTmp = boxA.u / boxA.u.norm(); checkAxis(axisTmp);
        axisTmp = boxA.v / boxA.v.norm(); checkAxis(axisTmp);
        axisTmp = boxA.w / boxA.w.norm(); checkAxis(axisTmp);

        // 2. tet face
        axisTmp = tetB.face(0).normal(); checkAxis(axisTmp);
        axisTmp = tetB.face(1).normal(); checkAxis(axisTmp);
        axisTmp = tetB.face(2).normal(); checkAxis(axisTmp);
        axisTmp = tetB.face(3).normal(); checkAxis(axisTmp);

        // 2. edge cross
        for (int i = 0; i < 3; i++)
        {
            Vec3f boxDir = (i == 0) ? (boxA.u) : ((i == 1) ? (boxA.v) : (boxA.w));
            for (int j = 0; j < 3; j++)
                for (int k = j + 1; k < 4; k++)
            {
                Vec3f tetDir = tetB.v[k] - tetB.v[j];
                axisTmp = boxDir.cross(tetDir);
                checkAxis(axisTmp);
            }
        }

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int j = 0; j < 4; j++)
        {
            Vec3f pB = tetB.v[j];
            Point3D queryP(pB);
            Point3D projP = queryP.project(boxA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }
    */


    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tetA, const OBox3D& boxB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on tetA
        MSDF(sat, boxB, tetA, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            sat.reverse();          // contact normal on boxB
            m.normal = sat.normal();

            setupContactOnBox(m, sat, tetA, boxB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& boxA, const Tet3D& tetB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on tetB
        MSDF(sat, boxA, tetB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on tetB

            setupContactOnTet(m, sat, boxA, tetB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }

    // ---------------------------------------- [Box - Box] ----------------------------------------
    
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(SeparationData& sat, const OBox3D& boxA, const OBox3D& boxB, const Real radiusA, const Real radiusB)
    {
        auto checkAxisP = [&](Vec3f pA, Vec3f pB) { checkAxisPoint(sat, boxA, boxB, radiusA, radiusB, pA, pB); };
        auto checkAxisE = [&](Segment3D edgeA, Segment3D edgeB) { checkAxisEdge(sat, boxA, boxB, radiusA, radiusB, edgeA, edgeB); };
        auto checkAxisR = [&](Rectangle3D face, auto type) { checkAxisRect(sat, boxA, boxB, radiusA, radiusB, face, type); };

        // Minkowski Face Normal
        // face
        // box face
        for (int j = 0; j < 6; j++) checkAxisR(boxA.face(j), CT_RECTA);
        for (int j = 0; j < 6; j++) checkAxisR(boxB.face(j), CT_RECTB);


        // Minkowski Edge-Edge Normal
        // boxA x boxB
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 12; j++)
            {
                checkAxisE(boxA.edge(i), boxB.edge(j));
            }

        // Minkowski Point-Point Normal
        for (int j = 0; j < 8; j++)
        {
            Vec3f pA = boxA.vertex(j).origin;
            Point3D queryP(pA);
            Point3D projP = queryP.project(boxB);
            checkAxisP(pA, projP.origin);
        }

        for (int j = 0; j < 8; j++)
        {
            Vec3f pB = boxB.vertex(j).origin;
            Point3D queryP(pB);
            Point3D projP = queryP.project(boxA);
            checkAxisP(projP.origin, pB);
        }
    }

    /*
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::MSDF(const OBox3D& boxA, const OBox3D& boxB, Real& depth, Coord3D& normal, Real& boundaryA, Real& boundaryB, const Real radiusA, const Real radiusB)
    {
        depth = -REAL_infinity;
        bool sign = true; // < 0
        auto checkAxis = [&](Vec3f N)
        {
            Real D = 0;
            Real bA, bB;
            if (N.norm() > EPSILON) N /= N.norm(); else return;
            checkSignedDistanceAxis(D, bA, bB, N, boxA, boxB, radiusA, radiusB);

            updateSDF(boundaryA, boundaryB, depth, normal, bA, bB, D, N);
        };
        Vec3f axisTmp;

        // Minkowski Face Normal
        // 1. BoxA face (u, v, w)
        axisTmp = boxA.u / boxA.u.norm(); checkAxis(axisTmp);
        axisTmp = boxA.v / boxA.v.norm(); checkAxis(axisTmp);
        axisTmp = boxA.w / boxA.w.norm(); checkAxis(axisTmp);

        // 2. BoxB face (u, v, w)
        axisTmp = boxB.u / boxB.u.norm(); checkAxis(axisTmp);
        axisTmp = boxB.v / boxB.v.norm(); checkAxis(axisTmp);
        axisTmp = boxB.w / boxB.w.norm(); checkAxis(axisTmp);

        // 3. boxA's edge x boxB's edge
        for (int i = 0; i < 3; i++)
        {
            Vec3f boxDirA = (i == 0) ? (boxA.u) : ((i == 1) ? (boxA.v) : (boxA.w));
            for (int j = 0; j < 3; j++)
            {
                Vec3f boxDirB = (j == 0) ? (boxB.u) : ((j == 1) ? (boxB.v) : (boxB.w));
                axisTmp = boxDirA.cross(boxDirB);
                checkAxis(axisTmp);
            }
        }

        // Minkowski Point-Edge Normal
        // check direction of minimum distance from B's point to A's edge
        for (int i = 0; i < 8; i++)
        {
            Vec3f pB = boxB.center  + ((i & 4) ? 1 : -1) * boxB.u * boxB.extent[0]
                                    + ((i & 2) ? 1 : -1) * boxB.v * boxB.extent[1]
                                    + ((i & 1) ? 1 : -1) * boxB.w * boxB.extent[2];
            Point3D queryP(pB);
            Point3D projP = queryP.project(boxA);
            axisTmp = (projP - queryP).direction();
            checkAxis(axisTmp);
        }
    }
    */

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& boxA, const OBox3D& boxB, const Real radiusA, const Real radiusB)
    {
        SeparationData sat;
        // Contact normal on boxB
        MSDF(sat, boxA, boxB, radiusA, radiusB);
        Real depth = sat.depth();

        if (REAL_LESS(depth, 0) && REAL_GREAT(depth, -REAL_infinity))
        {
            m.normal = sat.normal(); // contact normal on boxB

            setupContactOnBox(m, sat, boxA, boxB, radiusA, radiusB);
        }
        else m.contactCount = 0;
    }


    //--------------------------------------------------------------------------------------------------
    template<typename Real>
    DYN_FUNC void computeSupportEdge(Vector<Real, 3>& aOut, Vector<Real, 3>& bOut, const SquareMatrix<Real, 3>& rot, const Vector<Real, 3>& trans, const Vector<Real, 3>& e, Vector<Real, 3> n)
    {
        n = rot.transpose()*n;
        Vector<Real, 3> absN = abs(n);
        Vector<Real, 3> a, b;

        // x > y
        if (absN.x > absN.y)
        {
            // x > y > z
            if (absN.y > absN.z)
            {
                a = Vector<Real, 3>(e.x, e.y, e.z);
                b = Vector<Real, 3>(e.x, e.y, -e.z);
            }
            // x > z > y || z > x > y
            else
            {
                a = Vector<Real, 3>(e.x, e.y, e.z);
                b = Vector<Real, 3>(e.x, -e.y, e.z);
            }
        }

        // y > x
        else
        {
            // y > x > z
            if (absN.x > absN.z)
            {
                a = Vector<Real, 3>(e.x, e.y, e.z);
                b = Vector<Real, 3>(e.x, e.y, -e.z);
            }
            // z > y > x || y > z > x
            else
            {
                a = Vector<Real, 3>(e.x, e.y, e.z);
                b = Vector<Real, 3>(-e.x, e.y, e.z);
            }
        }

        Real signx = fsign(n.x);
        Real signy = fsign(n.y);
        Real signz = fsign(n.z);

        a.x *= signx;
        a.y *= signy;
        a.z *= signz;
        b.x *= signx;
        b.y *= signy;
        b.z *= signz;

        aOut = rot * a + trans;
        bOut = rot * b + trans;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D box0, const OBox3D box1)
    {
        m.contactCount = 0;

        Vector<Real, 3> v = box1.center - box0.center;

        Vector<Real, 3> eA = box0.extent;
        Vector<Real, 3> eB = box1.extent;

        Matrix3D rotA;
        rotA.setCol(0, box0.u);
        rotA.setCol(1, box0.v);
        rotA.setCol(2, box0.w);

        Matrix3D rotB;
        rotB.setCol(0, box1.u);
        rotB.setCol(1, box1.v);
        rotB.setCol(2, box1.w);

        // B's frame in A's space
        Matrix3D C = rotA.transpose() * rotB;
        Matrix3D absC;
        bool parallel = false;
        const float kCosTol = float(1.0e-6);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                float val = abs(C(i, j));
                absC(i, j) = val;

                if (val + kCosTol >= float(1.0))
                    parallel = true;
            }
        }

        Matrix3D C_t = C.transpose();
        Matrix3D absC_t = absC.transpose();

        // Vector from center A to center B in A's space
        Vector<Real, 3> t = rotA.transpose() * v;

        // Query states
        Real s;
        Real aMax = -REAL_MAX;
        Real bMax = -REAL_MAX;
        float eMax = -REAL_MAX;
        int aAxis = ~0;
        int bAxis = ~0;
        int eAxis = ~0;
        Vector<Real, 3> nA;
        Vector<Real, 3> nB;
        Vector<Real, 3> nE;

        // Face axis checks

        // a's x axis
        s = abs(t.x) - (box0.extent.x + absC_t.col(0).dot(box1.extent));
        if (trackFaceAxis(aAxis, aMax, nA, 0, s, rotA.col(0)))
            return;

        // a's y axis
        s = abs(t.y) - (box0.extent.y + absC_t.col(1).dot(box1.extent));
        if (trackFaceAxis(aAxis, aMax, nA, 1, s, rotA.col(1)))
            return;

        // a's z axis
        s = abs(t.z) - (box0.extent.z + absC_t.col(2).dot(box1.extent));
        if (trackFaceAxis(aAxis, aMax, nA, 2, s, rotA.col(2)))
            return;

        // b's x axis
        s = abs(t.dot(C.col(0))) - (box1.extent.x + absC.col(0).dot(box0.extent));
        if (trackFaceAxis(bAxis, bMax, nB, 3, s, rotB.col(0)))
            return;

        // b's y axis
        s = abs(t.dot(C.col(1))) - (box1.extent.y + absC.col(1).dot(box0.extent));
        if (trackFaceAxis(bAxis, bMax, nB, 4, s, rotB.col(1)))
            return;

        // b's z axis
        s = abs(t.dot(C.col(2))) - (box1.extent.z + absC.col(2).dot(box0.extent));
        if (trackFaceAxis(bAxis, bMax, nB, 5, s, rotB.col(2)))
            return;

        if (!parallel)
        {
            // Edge axis checks
            float rA;
            float rB;

            // Cross( a.x, b.x )
            rA = eA.y * absC(2, 0) + eA.z * absC(1, 0);
            rB = eB.y * absC(0, 2) + eB.z * absC(0, 1);
            s = abs(t.z * C(1, 0) - t.y * C(2, 0)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 6, s, Vector<Real, 3>(float(0.0), -C(2, 0), C(1, 0))))
                return;

            // Cross( a.x, b.y )
            rA = eA.y * absC(2, 1) + eA.z * absC(1, 1);
            rB = eB.x * absC(0, 2) + eB.z * absC(0, 0);
            s = abs(t.z * C(1, 1) - t.y * C(2, 1)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 7, s, Vector<Real, 3>(float(0.0), -C(2, 1), C(1, 1))))
                return;

            // Cross( a.x, b.z )
            rA = eA.y * absC(2, 2) + eA.z * absC(1, 2);
            rB = eB.x * absC(0, 1) + eB.y * absC(0, 0);
            s = abs(t.z * C(1, 2) - t.y * C(2, 2)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 8, s, Vector<Real, 3>(float(0.0), -C(2, 2), C(1, 2))))
                return;

            // Cross( a.y, b.x )
            rA = eA.x * absC(2, 0) + eA.z * absC(0, 0);
            rB = eB.y * absC(1, 2) + eB.z * absC(1, 1);
            s = abs(t.x * C(2, 0) - t.z * C(0, 0)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 9, s, Vector<Real, 3>(C(2, 0), float(0.0), -C(0, 0))))
                return;

            // Cross( a.y, b.y )
            rA = eA.x * absC(2, 1) + eA.z * absC(0, 1);
            rB = eB.x * absC(1, 2) + eB.z * absC(1, 0);
            s = abs(t.x * C(2, 1) - t.z * C(0, 1)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 10, s, Vector<Real, 3>(C(2, 1), float(0.0), -C(0, 1))))
                return;

            // Cross( a.y, b.z )
            rA = eA.x * absC(2, 2) + eA.z * absC(0, 2);
            rB = eB.x * absC(1, 1) + eB.y * absC(1, 0);
            s = abs(t.x * C(2, 2) - t.z * C(0, 2)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 11, s, Vector<Real, 3>(C(2, 2), float(0.0), -C(0, 2))))
                return;

            // Cross( a.z, b.x )
            rA = eA.x * absC(1, 0) + eA.y * absC(0, 0);
            rB = eB.y * absC(2, 2) + eB.z * absC(2, 1);
            s = abs(t.y * C(0, 0) - t.x * C(1, 0)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 12, s, Vector<Real, 3>(-C(1, 0), C(0, 0), float(0.0))))
                return;

            // Cross( a.z, b.y )
            rA = eA.x * absC(1, 1) + eA.y * absC(0, 1);
            rB = eB.x * absC(2, 2) + eB.z * absC(2, 0);
            s = abs(t.y * C(0, 1) - t.x * C(1, 1)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 13, s, Vector<Real, 3>(-C(1, 1), C(0, 1), float(0.0))))
                return;

            // Cross( a.z, b.z )
            rA = eA.x * absC(1, 2) + eA.y * absC(0, 2);
            rB = eB.x * absC(2, 1) + eB.y * absC(2, 0);
            s = abs(t.y * C(0, 2) - t.x * C(1, 2)) - (rA + rB);
            if (trackEdgeAxis(eAxis, eMax, nE, 14, s, Vector<Real, 3>(-C(1, 2), C(0, 2), float(0.0))))
                return;
        }

        // Artificial axis bias to improve frame coherence
        const float kRelTol = float(0.95);
        const float kAbsTol = float(0.01);
        int axis;
        float sMax;
        Vector<Real, 3> n;
        float faceMax = std::max(aMax, bMax);
        if (kRelTol * eMax > faceMax + kAbsTol)
        {
            axis = eAxis;
            sMax = eMax;
            n = nE;
        }
        else
        {
            if (kRelTol * bMax > aMax + kAbsTol)
            {
                axis = bAxis;
                sMax = bMax;
                n = nB;
            }
            else
            {
                axis = aAxis;
                sMax = aMax;
                n = nA;
            }
        }

        if (n.dot(v) < float(0.0))
            n = -n;

        if (axis == ~0)
            return;

        Transform3D atx(box0.center, rotA);
        Transform3D btx(box1.center, rotB);

        if (axis < 6)
        {
            Transform3D rtx;
            Transform3D itx;
            Vector<Real, 3> eR;
            Vector<Real, 3> eI;
            bool flip;

            if (axis < 3)
            {
                rtx = atx;
                itx = btx;
                eR = eA;
                eI = eB;
                flip = false;
            }
            else
            {
                rtx = btx;
                itx = atx;
                eR = eB;
                eI = eA;
                flip = true;
                n = -n;
            }

            // Compute reference and incident edge information necessary for clipping
            ClipVertex incident[4];
            computeIncidentFace(incident, itx, eI, n);
            unsigned char clipEdges[4];
            Matrix3D basis;
            Vector<Real, 3> e;
            computeReferenceEdgesAndBasis(clipEdges, &basis, &e, eR, rtx, n, axis);

            // Clip the incident face against the reference face side planes
            ClipVertex out[8];
            float depths[8];
            int outNum;
            outNum = clip(out, depths, rtx.translation(), e, clipEdges, basis, incident);

            if (outNum)
            {
                m.contactCount = outNum;
                m.normal = flip ? -n : n;

                for (int i = 0; i < outNum; ++i)
                {
                    m.contacts[i].position = flip ? (out[i].v + depths[i] * m.normal) : out[i].v;
                    m.contacts[i].penetration = depths[i];
                }
            }
        }
        else
        {
            n = rotA * n;

            if (n.dot(v) < float(0.0))
                n = -n;

            Vector<Real, 3> PA, QA;
            Vector<Real, 3> PB, QB;
            computeSupportEdge(PA, QA, rotA, box0.center, eA, n);
            computeSupportEdge(PB, QB, rotB, box1.center, eB, -n);

            Vector<Real, 3> CA, CB;
            edgesContact(CA, CB, PA, QA, PB, QB);

            m.normal = n;
            m.contactCount = 1;

            m.contacts[0].penetration = sMax;
            m.contacts[0].position = CB;
        }
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphere, const OBox3D& box)
    {
        m.contactCount = 0;

        Point3D c0(sphere.center);
        Real r0 = sphere.radius;

        Point3D vproj = c0.project(box);
        bool bInside = c0.inside(box);

        Segment3D dir = bInside ? c0 - vproj : vproj - c0;
        Real sMax = bInside ? -dir.direction().norm() - r0 : dir.direction().norm() - r0;

        if (sMax >= 0)
            return;

        m.normal = dir.direction().normalize();
        m.contacts[0].penetration = sMax;
        m.contacts[0].position = vproj.origin;
        m.contactCount = 1;
    }


    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Capsule3D& cap0, const Capsule3D& cap1)
    {
        m.contactCount = 0;

        Segment3D s0(cap0.centerline());
        Segment3D s1(cap1.centerline());
        Real r0 = cap0.radius + cap1.radius;

        // From cap0 to cap1
        Segment3D dir = s0.proximity(s1);

        dir = Point3D(dir.endPoint()) - Point3D(dir.startPoint());

        Real sMax = dir.direction().norm() - r0;
        if (sMax >= 0)
            return;

        m.normal = dir.direction().normalize();
        m.contacts[0].penetration = sMax;
        m.contacts[0].position = dir.v0 + (cap0.radius + sMax) * m.normal;
        m.contactCount = 1;
    }


    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphere, const Capsule3D& cap)
    {
        m.contactCount = 0;

        Point3D s(sphere.center);
        Segment3D c(cap.centerline());
        Real r0 = cap.radius + sphere.radius;

        Point3D pos = s.project(c);//pos: capsule

        // From sphere to capsule
        Segment3D dir = pos - s;

        Real sMax = dir.direction().norm() - r0;
        if (sMax >= 0)
            return;

        m.normal = dir.direction().normalize();
        m.contacts[0].penetration = sMax;
        m.contacts[0].position = dir.v0 + (sphere.radius + sMax) * m.normal;
        m.contactCount = 1;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Capsule3D& cap, const Sphere3D& sphere)
    {
        request(m, sphere, cap);

        swapContactPair(m);
    }

    template<typename Real>
    DYN_FUNC inline bool checkOverlapAxis(
        Real& lowerBoundary1,
        Real& upperBoundary1,
        Real& lowerBoundary2,
        Real& upperBoundary2,
        Real& intersectionDistance,
        Real& boundary1,
        Real& boundary2,
        const Vector<Real, 3> axisNormal,
        OrientedBox3D box,
        Capsule3D cap
        )
    {

        //projection to axis
        Segment3D seg = cap.centerline();

        lowerBoundary1 = seg.v0.dot(axisNormal) - cap.radius;
        upperBoundary1 = seg.v0.dot(axisNormal) + cap.radius;
        lowerBoundary1 = glm::min(lowerBoundary1, seg.v1.dot(axisNormal) - cap.radius);
        upperBoundary1 = glm::max(upperBoundary1, seg.v1.dot(axisNormal) + cap.radius);
        

        Vector<Real, 3> center = box.center;
        Vector<Real, 3> u = box.u;
        Vector<Real, 3> v = box.v;
        Vector<Real, 3> w = box.w;
        Vector<Real, 3> extent = box.extent;
        Vector<Real, 3> p;
        p = (center - u * extent[0] - v * extent[1] - w * extent[2]);
        lowerBoundary2 = upperBoundary2 = p.dot(axisNormal);

        p = (center - u * extent[0] - v * extent[1] + w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center - u * extent[0] + v * extent[1] - w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center - u * extent[0] + v * extent[1] + w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center + u * extent[0] - v * extent[1] - w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center + u * extent[0] - v * extent[1] + w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center + u * extent[0] + v * extent[1] - w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center + u * extent[0] + v * extent[1] + w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        return checkOverlap(lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2, intersectionDistance, boundary1, boundary2);
    }

    template<typename Real>
    DYN_FUNC inline bool checkOverlapTetTri(
        Real lowerBoundary1,
        Real upperBoundary1,
        Real lowerBoundary2,
        Real upperBoundary2,
        Real& intersectionDistance,
        Real& boundary1,
        Real& boundary2
    )
    {
        if (!((lowerBoundary1 > upperBoundary2) || (lowerBoundary2 > upperBoundary1)))
        {
            if (lowerBoundary1 < lowerBoundary2)
            {
                if (upperBoundary1 > upperBoundary2)
                {
                    if (upperBoundary2 - lowerBoundary1 > upperBoundary1 - lowerBoundary2)
                    {
                        boundary1 = upperBoundary1;
                        boundary2 = lowerBoundary2;
                        intersectionDistance = abs(boundary1 - boundary2);
                    }
                    else
                    {
                        boundary1 = lowerBoundary1;
                        boundary2 = upperBoundary2;
                        intersectionDistance = abs(boundary1 - boundary2);
                    }
                }
                else
                {
                    intersectionDistance = upperBoundary1 - lowerBoundary2;
                    boundary1 = upperBoundary1;
                    boundary2 = lowerBoundary2;
                }
            }
            else
            {
                if (upperBoundary1 > upperBoundary2)
                {
                    intersectionDistance = upperBoundary2 - lowerBoundary1;
                    boundary1 = lowerBoundary1;
                    boundary2 = upperBoundary2;
                }
                else
                {
                    //intersectionDistance = upperBoundary1 - lowerBoundary1;
                    if (upperBoundary2 - lowerBoundary1 > upperBoundary1 - lowerBoundary2)
                    {
                        boundary1 = upperBoundary1;
                        boundary2 = lowerBoundary2;
                        intersectionDistance = abs(boundary1 - boundary2);
                    }
                    else
                    {
                        boundary1 = lowerBoundary1;
                        boundary2 = upperBoundary2;
                        intersectionDistance = abs(boundary1 - boundary2);
                    }
                }
            }
            return true;
        }
        intersectionDistance = Real(0.0f);
        return false;
    }

    template<typename Real>
    DYN_FUNC inline bool checkOverlapAxis(
        Real& lowerBoundary1,
        Real& upperBoundary1,
        Real& lowerBoundary2,
        Real& upperBoundary2,
        Real& intersectionDistance,
        Real& boundary1,
        Real& boundary2,
        const Vector<Real, 3> axisNormal,
        Tet3D tet,
        Capsule3D cap)
    {

        Segment3D seg = cap.centerline();
        //projection to axis

        lowerBoundary1 = seg.v0.dot(axisNormal) - cap.radius;
        upperBoundary1 = seg.v0.dot(axisNormal) + cap.radius;
        lowerBoundary1 = glm::min(lowerBoundary1, seg.v1.dot(axisNormal) - cap.radius);
        upperBoundary1 = glm::max(upperBoundary1, seg.v1.dot(axisNormal) + cap.radius);

        for (int i = 0; i < 4; i++)
        {
            if (i == 0)
            {
                
                lowerBoundary2 = upperBoundary2 = tet.v[0].dot(axisNormal);
            }
            else
            {
                lowerBoundary2 = glm::min(lowerBoundary2, tet.v[i].dot(axisNormal));
                upperBoundary2 = glm::max(upperBoundary2, tet.v[i].dot(axisNormal));
            }
        }

        
        return checkOverlapTetTri(lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2, intersectionDistance, boundary1, boundary2);
    }


    template<typename Real>
    DYN_FUNC inline void setupContactCaps(
        Real boundary1,
        Real boundary2,
        const Vector<Real, 3> axisNormal,
        TCapsule3D<Real> cap,
        TOrientedBox3D<Real> box,
        Real depth,
        TManifold<Real>& m)
    {
        // Gen contact point on box
        // On box vertex

        // On box edge
        
        // On box face

        int cnt1, cnt2;
        Vector<Real, 3> boundaryPoints1[4]; 
        Vector<Real, 3> boundaryPoints2[8];
        cnt1 = cnt2 = 0;

        Real boundaryMin = glm::min(boundary1, boundary2);
        Real boundaryMax = glm::max(boundary1, boundary2);
        if (abs(cap.startPoint().dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints1[cnt1++] = cap.startPoint();
        if (abs(cap.endPoint().dot(axisNormal)  - boundaryMin) < abs(depth))
            boundaryPoints1[cnt1++] = cap.endPoint();
        

        Vector<Real, 3> center = box.center;
        Vector<Real, 3> u = box.u;
        Vector<Real, 3> v = box.v;
        Vector<Real, 3> w = box.w;
        Vector<Real, 3> extent = box.extent;
        Vector<Real, 3> p;
        p = (center - u * extent[0] - v * extent[1] - w * extent[2]);
        if (abs(p.dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints2[cnt2++] = p;

        p = (center - u * extent[0] - v * extent[1] + w * extent[2]);
        if (abs(p.dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints2[cnt2++] = p;

        p = (center - u * extent[0] + v * extent[1] - w * extent[2]);
        if (abs(p.dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints2[cnt2++] = p;

        p = (center - u * extent[0] + v * extent[1] + w * extent[2]);
        if (abs(p.dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints2[cnt2++] = p;

        p = (center + u * extent[0] - v * extent[1] - w * extent[2]);
        if (abs(p.dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints2[cnt2++] = p;

        p = (center + u * extent[0] - v * extent[1] + w * extent[2]);
        if (abs(p.dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints2[cnt2++] = p;

        p = (center + u * extent[0] + v * extent[1] - w * extent[2]);
        if (abs(p.dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints2[cnt2++] = p;

        p = (center + u * extent[0] + v * extent[1] + w * extent[2]);
        if (abs(p.dot(axisNormal) - boundaryMin) < abs(depth))
            boundaryPoints2[cnt2++] = p;

        // printf("cnt1 = %d, cnt2 = %d,  dep=%.3lf\n", cnt1, cnt2, depth);
        if (cnt1 == 1 || cnt2 == 1)
        {
            m.normal = (boundary1 < boundary2) ? -axisNormal : axisNormal;
            m.contacts[0].penetration = depth;
            m.contacts[0].position = (cnt1 == 1) ? boundaryPoints1[0] : boundaryPoints2[0];
            m.contactCount = 1;
            return;
        }
        else if (cnt1 == 2)
        {
            Segment3D s1 = cap.centerline();

            if (cnt2 == 2)
            {
                Segment3D s2(boundaryPoints2[0], boundaryPoints2[1]);
                Segment3D dir = s1.proximity(s2);//v0: self v1: other
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                m.contacts[0].penetration = depth;
                m.contacts[0].position = dir.v0;
                m.contactCount = 1;
                return;
            }
            else //cnt2 == 4
            {
                //if (cnt2 != 4)
                //	printf("?????????\n");


                for (int tmp_i = 1; tmp_i < 4; tmp_i++)
                    for (int tmp_j = tmp_i + 1; tmp_j < 4; tmp_j++)
                    {
                        if ((boundaryPoints2[tmp_i] - boundaryPoints2[0]).dot(boundaryPoints2[tmp_j] - boundaryPoints2[0]) < EPSILON)
                        {
                            int tmp_k = 1 + 2 + 3 - tmp_i - tmp_j;
                            Vector<Real, 3> p2 = boundaryPoints2[tmp_i];
                            Vector<Real, 3> p3 = boundaryPoints2[tmp_j];
                            Vector<Real, 3> p4 = boundaryPoints2[tmp_k];
                            boundaryPoints2[1] = p2;
                            boundaryPoints2[2] = p3;
                            boundaryPoints2[3] = p4;
                            break;
                        }
                    }

                //printf("%.3lf %.3lf %.3lf\n", boundaryPoints2[0][0], boundaryPoints2[0][1], boundaryPoints2[0][2]);
                //printf("%.3lf %.3lf %.3lf\n", boundaryPoints2[1][0], boundaryPoints2[1][1], boundaryPoints2[1][2]);
                //printf("%.3lf %.3lf %.3lf\n", boundaryPoints2[2][0], boundaryPoints2[2][1], boundaryPoints2[2][2]);
                //printf("%.3lf %.3lf %.3lf\n", boundaryPoints2[3][0], boundaryPoints2[3][1], boundaryPoints2[3][2]);

                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                Triangle3D t2(boundaryPoints2[0], boundaryPoints2[1], boundaryPoints2[2]);
                Vector<Real, 3> dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
                Vector<Real, 3> dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;

                //printf("%.3lf %.3lf %.3lf\n", axisNormal[0], axisNormal[1], axisNormal[2]);
                //printf("%.3lf %.3lf %.3lf %.5lf %.5lf %.5lf\n", dirTmp1[0], dirTmp1[1], dirTmp1[2], dirTmp1.cross(axisNormal)[0], dirTmp1.cross(axisNormal)[1], dirTmp1.cross(axisNormal)[2]);
                //printf("%.3lf %.3lf %.3lf %.5lf %.5lf %.5lf\n", dirTmp2[0], dirTmp2[1], dirTmp2[2], dirTmp2.cross(axisNormal)[0], dirTmp2.cross(axisNormal)[1], dirTmp2.cross(axisNormal)[2]);


                if (dirTmp1.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = depth;
                    m.contacts[m.contactCount].position = s1.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = depth;
                    m.contacts[m.contactCount].position = s1.v1;
                    m.contactCount++;
                }
                t2 = Triangle3D(boundaryPoints2[3], boundaryPoints2[1], boundaryPoints2[2]);
                dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
                dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;

                //printf("%.3lf %.3lf %.3lf\n", dirTmp1[0], dirTmp1[1], dirTmp1[2]);
                //printf("%.3lf %.3lf %.3lf\n", dirTmp2[0], dirTmp2[1], dirTmp2[2]);

                if (dirTmp1.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = depth;
                    m.contacts[m.contactCount].position = s1.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = depth;
                    m.contacts[m.contactCount].position = s1.v1;
                    m.contactCount++;
                }
                for (int i = 0; i < 4; i++)
                {
                    Segment3D s2;
                    if (i < 2)
                        s2 = Segment3D(boundaryPoints2[0], boundaryPoints2[i]);
                    else
                        s2 = Segment3D(boundaryPoints2[3], boundaryPoints2[i - 2]);
                    Segment3D dir = s1.proximity(s2);
                    //printf("dir: %.3lf %.3lf %.3lf\naxisnormal %.3lf %.3lf %.3lf\n%.6lf\n",
                    //	dir.direction()[0], dir.direction()[1], dir.direction()[2],
                    //	axisNormal[0], axisNormal[1], axisNormal[2],
                    //	dir.direction().normalize().cross(axisNormal).norm());
                    if ((!dir.isValid()) || dir.direction().normalize().cross(axisNormal).norm() < 1e-4)
                    {
                        //printf("Yes\n");
                        if ((dir.v0 - s1.v0).norm() > 1e-4 && (dir.v0 - s1.v1).norm() > 1e-4)
                        {
                            m.contacts[m.contactCount].penetration = depth;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                        }
                    }
                }
            }
        }
        
    }

    template<typename Real>
    DYN_FUNC inline void setupContactTets(
        Real boundary1,
        Real boundary2,
        const Vector<Real, 3> axisNormal,
        Tet3D tet,
        Capsule3D cap,
        Real sMax,
        TManifold<Real>& m)
    {
        int cnt1, cnt2;
        unsigned char boundaryPoints1[4], boundaryPoints2[4];
        cnt1 = cnt2 = 0;



        for (unsigned char i = 0; i < 4; i++)
        {
            if (abs(tet.v[i].dot(axisNormal) - boundary1) < abs(sMax))
                boundaryPoints1[cnt1 ++] = i;
        }

        if (abs(cap.startPoint().dot(axisNormal) + cap.radius - boundary1) < abs(sMax)
            ||
            abs(cap.startPoint().dot(axisNormal) - cap.radius - boundary1) < abs(sMax))
            boundaryPoints2[cnt2 ++] = 0;
        
        if (abs(cap.endPoint().dot(axisNormal) + cap.radius - boundary1) < abs(sMax)
            ||
            abs(cap.endPoint().dot(axisNormal) - cap.radius - boundary1) < abs(sMax))
            boundaryPoints2[cnt2++] = 1;

        if (cnt1 == 1)
        {
            m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
            m.contacts[0].penetration = sMax;
            m.contacts[0].position = tet.v[boundaryPoints1[0]];
            m.contactCount = 1;
            return;
        }
        if (cnt2 == 1)
        {
            m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
            m.contacts[0].penetration = sMax;
            m.contacts[0].position = boundaryPoints2[0] ? cap.endPoint() : cap.startPoint();
            m.contactCount = 1;
            return;
        }
        else if (cnt1 == 2)
        {
            Segment3D s1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]]);

            //if (cnt2 == 2)
            {
                Segment3D s2 = cap.centerline();
                Segment3D dir = s1.proximity(s2);//v0: self v1: other
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                m.contacts[0].penetration = sMax;
                m.contacts[0].position = dir.v0;
                m.contactCount = 1;
                return;
            }
        }
        else if (cnt1 == 3)
        {
            Triangle3D t1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]], tet.v[boundaryPoints1[2]]);
            //if (cnt2 == 2)
            {

                Segment3D s2 = cap.centerline();
                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;

                Vector<Real, 3> dirTmp1 = Point3D(s2.v0).project(t1).origin - s2.v0;
                Vector<Real, 3> dirTmp2 = Point3D(s2.v1).project(t1).origin - s2.v1;
                if (dirTmp1.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s2.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s2.v1;
                    m.contactCount++;
                }
                for (int i = 0; i < 3; i++)
                {
                    Segment3D s1(t1.v[(i + 1) % 3], t1.v[(i + 2) % 3]);
                    Segment3D dir = s2.proximity(s1);
                    if ((!dir.isValid()) || dir.direction().normalize().cross(axisNormal).norm() < 1e-5)
                    {
                        if ((dir.v0 - s2.v0).norm() > 1e-5 && (dir.v0 - s2.v1).norm() > 1e-5)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                        }
                    }
                }
            }
            
        }
    }


    template<typename Real>
    DYN_FUNC inline void setupContactTetTri(
        Real boundary1,
        Real boundary2,
        const Vector<Real, 3> axisNormal,
        Tet3D tet,
        Triangle3D triangle,
        Real sMax,
        TManifold<Real>& m)
    {
        int cnt1, cnt2;
        unsigned char boundaryPoints1[4], boundaryPoints2[4];
        cnt1 = cnt2 = 0;



        for (unsigned char i = 0; i < 4; i++)
        {
            if (abs(tet.v[i].dot(axisNormal) - boundary1) < abs(sMax) + EPSILON)
                boundaryPoints1[cnt1++] = i;
        }

        for (unsigned char i = 0; i < 3; i++)
        {
            if (abs(triangle.v[i].dot(axisNormal) - boundary2) < abs(sMax) + EPSILON)
                boundaryPoints2[cnt2++] = i;
        }

        //printf("cnt1 = %d   cnt2 = %d  smax = %.10lf\n", cnt1, cnt2, sMax);

        if (cnt1 == 1 || cnt2 == 1)
        {
            m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
            m.contacts[0].penetration = sMax;
            m.contacts[0].position = (cnt1 == 1) ? tet.v[boundaryPoints1[0]] : triangle.v[boundaryPoints2[0]];
            m.contactCount = 1;
            return;
        }
        else if (cnt1 == 2)
        {
            Segment3D s1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]]);

            if (cnt2 == 2)
            {
                Segment3D s2(triangle.v[boundaryPoints2[0]], triangle.v[boundaryPoints2[1]]);
                Segment3D dir = s1.proximity(s2);//v0: self v1: other
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                m.contacts[0].penetration = sMax;
                m.contacts[0].position = dir.v0;
                m.contactCount = 1;
                return;
            }
            else //cnt2 == 3
            {
                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                Triangle3D t2(triangle.v[boundaryPoints2[0]], triangle.v[boundaryPoints2[1]], triangle.v[boundaryPoints2[2]]);
                Vector<Real, 3> dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
                Vector<Real, 3> dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;
                if (dirTmp1.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s1.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s1.v1;
                    m.contactCount++;
                }
                for (int i = 0; i < 3; i++)
                {
                    Segment3D s2(t2.v[(i + 1) % 3], t2.v[(i + 2) % 3]);
                    Segment3D dir = s1.proximity(s2);
                    
                    if ((!dir.isValid()) || dir.direction().normalize().cross(axisNormal).norm() < 1e-5)
                    {
                        //printf("Yes\n");
                        if ((dir.v0 - s1.v0).norm() > 1e-5 && (dir.v0 - s1.v1).norm() > 1e-5)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                        }
                    }
                }
            }
        }
        else if (cnt1 == 3)
        {
            Triangle3D t1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]], tet.v[boundaryPoints1[2]]);
            if (cnt2 == 2)
            {

                Segment3D s2(triangle.v[boundaryPoints2[0]], triangle.v[boundaryPoints2[1]]);
                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;

                Vector<Real, 3> dirTmp1 = Point3D(s2.v0).project(t1).origin - s2.v0;
                Vector<Real, 3> dirTmp2 = Point3D(s2.v1).project(t1).origin - s2.v1;
                if (dirTmp1.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s2.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s2.v1;
                    m.contactCount++;
                }
                for (int i = 0; i < 3; i++)
                {
                    Segment3D s1(t1.v[(i + 1) % 3], t1.v[(i + 2) % 3]);
                    Segment3D dir = s2.proximity(s1);
                    if ((!dir.isValid()) || dir.direction().normalize().cross(axisNormal).norm() < 1e-5)
                    {
                        if ((dir.v0 - s2.v0).norm() > 1e-5 && (dir.v0 - s2.v1).norm() > 1e-5)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                        }
                    }
                }
            }
            if (cnt2 == 3)
            {
                Triangle3D t1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]], tet.v[boundaryPoints1[2]]);
                Triangle3D t2 = triangle;

                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;

                for (int i = 0; i < 3; i++)
                {
                    if ((Point3D(t1.v[i]).project(t2).origin - t1.v[i]).cross(t2.normal()).norm() < 1e-5)
                    {
                        m.contacts[m.contactCount].penetration = sMax;
                        m.contacts[m.contactCount].position = t1.v[i];
                        m.contactCount++;
                    }
                    if ((Point3D(t2.v[i]).project(t1).origin - t2.v[i]).cross(t1.normal()).norm() < 1e-5)
                    {
                        m.contacts[m.contactCount].penetration = sMax;
                        m.contacts[m.contactCount].position = t2.v[i];
                        m.contactCount++;
                    }

                    for (int j = 0; j < 3; j++)
                    {
                        Segment3D s1(t1.v[(i + 1) % 3], t1.v[(i + 2) % 3]);
                        Segment3D s2(t2.v[(j + 1) % 3], t2.v[(j + 2) % 3]);
                        Segment3D dir = s1.proximity(s2);
                        if ((!dir.isValid()) || dir.direction().normalize().cross(axisNormal).norm() < 1e-5)
                        {
                            if ((dir.v0 - s1.v0).norm() > 1e-5 && (dir.v0 - s1.v1).norm() > 1e-5)
                            {
                                m.contacts[m.contactCount].penetration = sMax;
                                m.contacts[m.contactCount].position = dir.v0;
                                m.contactCount++;
                            }
                        }
                    }
                }

            }
        }
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Capsule3D& cap, const OBox3D& box)
    {
        Segment3D c(cap.centerline());
        Real r0 = cap.radius;

        m.contactCount = 0;


        Segment3D s0(cap.centerline());
        Segment3D ints = s0;
        
        int segInter = s0.intersect(box, ints);
        
        bool bInter = false;

        // Check if intersection
        if (segInter == 0)
        {
            Segment3D dir = s0.proximity(box);
            Real sMax = dir.direction().norm() - r0;
            //printf("Capsule 2 Box Broad: %.4f\n", sMax);
            if (sMax < 0) bInter = true;
        }
        else
        {
            bInter = true;
        }

        if (!bInter) return;

        // like SAT (Select collision direction)
        { 
            Real sMax = (Real)INT_MAX;
            Real sIntersect; // >0
            Real lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2;
            Real l1, u1, l2, u2;
            Vector<Real, 3> axis = Vector<Real, 3>(0, 1, 0);
            Vector<Real, 3> axisTmp = axis;

            Real boundary1, boundary2, b1, b2;

            //u
            axisTmp = box.u / box.u.norm();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, box, cap) == false)
            {
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }


            //v
            axisTmp = box.v / box.v.norm();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, box, cap) == false)
            {
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }

            //w
            axisTmp = box.w / box.w.norm();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, box, cap) == false)
            {
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }
            
            //dir generated by cross product from capsule and box
            Vector<Real, 3> dirCap = c.direction();
            for (int j = 0; j < 3; j++)
            {
                Vector<Real, 3> boxDir = (j == 0) ? (box.u) : ((j == 1) ? (box.v) : (box.w));
                axisTmp = dirCap.cross(boxDir);
                if (axisTmp.norm() > EPSILON)
                {
                    axisTmp /= axisTmp.norm();
                }
                else //parallel, choose an arbitary direction
                {
                    if (abs(dirCap[0]) > EPSILON)
                        axisTmp = Vector<Real, 3>(dirCap[1], -dirCap[0], 0);
                    else
                        axisTmp = Vector<Real, 3>(0, dirCap[2], -dirCap[1]);
                    axisTmp /= axisTmp.norm();
                }
                if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, box, cap) == false)
                {
                    m.contactCount = 0;
                    return;
                }
                else
                {
                    if (sIntersect < sMax)
                    {
                        sMax = sIntersect;
                        lowerBoundary1 = l1;
                        lowerBoundary2 = l2;
                        upperBoundary1 = u1;
                        upperBoundary2 = u2;
                        boundary1 = b1;
                        boundary2 = b2;
                        axis = axisTmp;
                    }
                }
            }


            setupContactCaps(boundary1, boundary2, axis, cap, box, -sMax, m);

            //for (uint i = 0; i < m.contactCount; i++)
            //{
            //	m.contacts[i].position += (cap.radius + m.contacts[i].penetration) * m.normal;
            //}
        }
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& box, const Capsule3D& cap)
    {
        request(m, cap, box);
        
        swapContactPair(m);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tet, const Triangle3D& tri)
    {
        m.contactCount = 0;

        Real sMax = (Real)INT_MAX;
        Real sIntersect;
        Real lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2;
        Real l1, u1, l2, u2;
        Vector<Real, 3> axis = Vector<Real, 3>(0, 1, 0);
        Vector<Real, 3> axisTmp = axis;

        Real boundary1, boundary2, b1, b2;

    

        for (int i = 0; i < 4; i++)
        {
            //tet face axis i
            axisTmp = tet.face(i).normal();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, tri) == false)
            {
                //printf("aax\n");
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }
        }


        //triangle face normal
        axisTmp = tri.normal();
        if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, tri) == false)
        {
            m.contactCount = 0;
            //printf("bbx\n");
            return;
        }
        else
        {
            if (sIntersect < sMax)
            {
                sMax = sIntersect;
                lowerBoundary1 = l1;
                lowerBoundary2 = l2;
                upperBoundary1 = u1;
                upperBoundary2 = u2;
                boundary1 = b1;
                boundary2 = b2;
                axis = axisTmp;
            }
        }

        const int segmentIndex[6][2] = {
        0, 1,
        0, 2,
        0, 3,
        1, 2,
        1, 3,
        2, 3
        };

        const int triIndex[6][2] = {
        0, 1,
        0, 2,
        1, 2
        };

        for (int i = 0; i < 6; i++)
            for(int j = 0; j < 3; j ++)
            {
                Vector<Real, 3> dirTet = tet.v[segmentIndex[i][0]] - tet.v[segmentIndex[i][1]];
                Vector<Real, 3> dirTri = tri.v[triIndex[j][0]] - tri.v[triIndex[j][1]];
                dirTri /= dirTri.norm();
                dirTet /= dirTet.norm();
                axisTmp = dirTet.cross(dirTri);
                if (axisTmp.norm() > EPSILON)
                {
                    axisTmp /= axisTmp.norm();
                }
                else //parallel, choose an arbitary direction
                {
                    if (abs(dirTet[0]) > EPSILON)
                        axisTmp = Vector<Real, 3>(dirTet[1], -dirTet[0], 0);
                    else
                        axisTmp = Vector<Real, 3>(0, dirTet[2], -dirTet[1]);
                    axisTmp /= axisTmp.norm();
                }
                if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, tri) == false)
                {
                    m.contactCount = 0;
                    //printf("ccx\n");
                    return;
                }
                else
                {
                    if (sIntersect < sMax)
                    {
                        sMax = sIntersect;
                        lowerBoundary1 = l1;
                        lowerBoundary2 = l2;
                        upperBoundary1 = u1;
                        upperBoundary2 = u2;
                        boundary1 = b1;
                        boundary2 = b2;
                        axis = axisTmp;
                    }
                }
            }

        axisTmp = (tet.v[0] + tet.v[1] + tet.v[2] + tet.v[3]) / 4.0f - (tri.v[0] + tri.v[1] + tri.v[2]) / 3.0f;
        axisTmp /= axisTmp.norm();
        if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, tri) == false)
        {
            m.contactCount = 0;
            //printf("ccx\n");
            return;
        }
        else
        {
            if (sIntersect < sMax)
            {
                sMax = sIntersect;
                lowerBoundary1 = l1;
                lowerBoundary2 = l2;
                upperBoundary1 = u1;
                upperBoundary2 = u2;
                boundary1 = b1;
                boundary2 = b2;
                axis = axisTmp;
            }
        }

        //printf("YesYes\n");
        setupContactTetTri(boundary1, boundary2, axis, tet, tri, -sMax, m);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Triangle3D& tri, const Tet3D& tet)
    {
        request(m, tet, tri);
        m.normal = -m.normal;
    }

    //Separating Axis Theorem for tets
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Capsule3D& cap, const Tet3D& tet)
    {
        m.contactCount = 0;

        Real sMax = (Real)INT_MAX;
        Real sIntersect;
        Real lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2;
        Real l1, u1, l2, u2;
        Vector<Real, 3> axis = Vector<Real, 3>(0, 1, 0);
        Vector<Real, 3> axisTmp = axis;

        Real boundary1, boundary2, b1, b2;


        for (int i = 0; i < 4; i++)
        {
            //tet0 face axis i
            axisTmp = tet.face(i).normal();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, cap) == false)
            {
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }
            //tet1 face axis i
            axisTmp = tet.face(i).normal();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, cap) == false)
            {
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }
        }

        const int segmentIndex[6][2] = {
        0, 1,
        0, 2,
        0, 3,
        1, 2,
        1, 3,
        2, 3
        };

        axisTmp = cap.centerline().direction();
        if (axisTmp.norm() > EPSILON)
        {
            axisTmp /= axisTmp.norm();
        }
        if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, cap) == false)
        {
            m.contactCount = 0;
            return;
        }
        else
        {
            if (sIntersect < sMax)
            {
                sMax = sIntersect;
                lowerBoundary1 = l1;
                lowerBoundary2 = l2;
                upperBoundary1 = u1;
                upperBoundary2 = u2;
                boundary1 = b1;
                boundary2 = b2;
                axis = axisTmp;
            }
        }

        for (int i = 0; i < 6; i++)
        {
            Vector<Real, 3> dirTet = tet.v[segmentIndex[i][0]] - tet.v[segmentIndex[i][1]];
            Vector<Real, 3> dirCap = cap.centerline().direction();

            

            axisTmp = dirTet.cross(dirCap);
            if (axisTmp.norm() > EPSILON)
            {
                axisTmp /= axisTmp.norm();
            }
            else //parallel, choose an arbitary direction
            {
                if (abs(dirTet[0]) > EPSILON)
                    axisTmp = Vector<Real, 3>(dirTet[1], -dirTet[0], 0);
                else
                    axisTmp = Vector<Real, 3>(0, dirTet[2], -dirTet[1]);
                axisTmp /= axisTmp.norm();
            }
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, cap) == false)
            {
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }
        }
        
        setupContactTets(boundary1, boundary2, axis, tet, cap, -sMax, m);

        for (uint i = 0; i < m.contactCount; i++)
        {
            m.contacts[i].position += (cap.radius + m.contacts[i].penetration) * m.normal;
        }
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tet, const Capsule3D& cap)
    {
        request(m, cap, tet);
        
        swapContactPair(m);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& box, const Sphere3D& sphere)
    {
        request(m, sphere, box);

        swapContactPair(m);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphere0, const Sphere3D& sphere1)
    {
        m.contactCount = 0;

        auto c0 = sphere0.center;
        auto c1 = sphere1.center;

        Real r0 = sphere0.radius;
        Real r1 = sphere1.radius;

        auto dir = c1 - c0;
        Real sMax = dir.norm() - r0 - r1;
        if (sMax >= 0)
            return;

        m.normal = dir.normalize();
        m.contacts[0].penetration = sMax;
        m.contacts[0].position = c0 + (r0 + sMax) * m.normal;
        m.contactCount = 1;
    }

    template<typename Real>
    DYN_FUNC inline bool checkOverlap(
        Real lowerBoundary1, 
        Real upperBoundary1, // A
        Real lowerBoundary2,
        Real upperBoundary2, // B
        Real& intersectionDistance,
        Real& boundary1,
        Real& boundary2
    )
    {
        if (!((lowerBoundary1 > upperBoundary2) || (lowerBoundary2 > upperBoundary1)))
        {
            if (lowerBoundary1 < lowerBoundary2)
            {
                if (upperBoundary1 > upperBoundary2)
                {
                    //     |---B---|
                    //   |-----A-----|
                    //intersectionDistance = upperBoundary2 - lowerBoundary2;
                    if (upperBoundary2 - lowerBoundary1 > upperBoundary1 - lowerBoundary2)
                    {
                        //      |---B---|(->)
                        //   |-----A-----|
                        boundary1 = upperBoundary1;
                        boundary2 = lowerBoundary2;
                        intersectionDistance = upperBoundary1 - lowerBoundary2;
                    }
                    else
                    {
                        // (<-)|---B---|
                        //    |-----A-----|
                        boundary1 = lowerBoundary1;
                        boundary2 = upperBoundary2;
                        intersectionDistance = upperBoundary2 - lowerBoundary1;
                    }
                }
                else
                {
                    //	    |---B---|(->)
                    //   |----A----|
                    intersectionDistance = upperBoundary1 - lowerBoundary2;
                    boundary1 = upperBoundary1;
                    boundary2 = lowerBoundary2;
                }
            }
            else
            {
                if (upperBoundary1 > upperBoundary2)
                {
                    //	(<-)|---B---|
                    //        |----A----|
                    intersectionDistance = upperBoundary2 - lowerBoundary1;
                    boundary1 = lowerBoundary1;
                    boundary2 = upperBoundary2;
                }
                else
                {
                    //     |-----B------|
                    //        |---A---|
                    //intersectionDistance = upperBoundary1 - lowerBoundary1;
                    if (upperBoundary2 - lowerBoundary1 > upperBoundary1 - lowerBoundary2)
                    {
                        //	   |-----B-----|(->)
                        //      |---A---|
                        boundary1 = upperBoundary1;
                        boundary2 = lowerBoundary2;
                        intersectionDistance = upperBoundary1 - lowerBoundary2;
                    }
                    else
                    {
                        //	   (<-)|------B------|
                        //              |---A---|
                        boundary1 = lowerBoundary1;
                        boundary2 = upperBoundary2;
                        intersectionDistance = upperBoundary2 - lowerBoundary1;
                    }
                }
            }
            return true;
        }
        // |---A---| |---B---|
        // |---B---| |---A---|
        intersectionDistance = Real(0.0f);
        return false;
    }

    template<typename Real>
    DYN_FUNC inline bool checkOverlapAxis(
        Real& lowerBoundary1,
        Real& upperBoundary1,
        Real& lowerBoundary2,
        Real& upperBoundary2,
        Real& intersectionDistance,
        Real& boundary1,
        Real& boundary2,
        const Vector<Real, 3> axisNormal,
        Tet3D tet1,
        Tet3D tet2)
    {

        //projection to axis
        for (int i = 0; i < 4; i++)
        {
            if (i == 0)
            {
                lowerBoundary1 = upperBoundary1 = tet1.v[0].dot(axisNormal);
                lowerBoundary2 = upperBoundary2 = tet2.v[0].dot(axisNormal);
            }
            else
            {
                lowerBoundary1 = glm::min(lowerBoundary1, tet1.v[i].dot(axisNormal));
                lowerBoundary2 = glm::min(lowerBoundary2, tet2.v[i].dot(axisNormal));
                upperBoundary1 = glm::max(upperBoundary1, tet1.v[i].dot(axisNormal));
                upperBoundary2 = glm::max(upperBoundary2, tet2.v[i].dot(axisNormal));
            }
        }

        /*printf(" axis = %.3lf  %.3lf  %.3lf\nlb1 = %.3lf lb2 = %.3lf\nub1 = %.3lf ub2 = %.3lf\n", 
            axisNormal[0], axisNormal[1], axisNormal[2],
            lowerBoundary1, lowerBoundary2,
            upperBoundary1, upperBoundary2
            );*/
        return checkOverlap(lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2, intersectionDistance, boundary1, boundary2);
    }


    template<typename Real>
    DYN_FUNC inline bool checkOverlapAxis(
        Real& lowerBoundary1,
        Real& upperBoundary1,
        Real& lowerBoundary2,
        Real& upperBoundary2,
        Real& intersectionDistance,
        Real& boundary1,
        Real& boundary2,
        const Vector<Real, 3> axisNormal,
        Tet3D tet,
        Triangle3D tri)
    {

        //projection to axis
        for (int i = 0; i < 4; i++)
        {
            if (i == 0)
            {
                lowerBoundary1 = upperBoundary1 = tet.v[0].dot(axisNormal);
            }
            else
            {
                lowerBoundary1 = glm::min(lowerBoundary1, tet.v[i].dot(axisNormal));
                upperBoundary1 = glm::max(upperBoundary1, tet.v[i].dot(axisNormal));
            }
        }

        for (int i = 0; i < 3; i++)
        {
            if(i == 0)
                lowerBoundary2 = upperBoundary2 = tri.v[i].dot(axisNormal);
            else
            {
                lowerBoundary2 = glm::min(lowerBoundary2, tri.v[i].dot(axisNormal));
                upperBoundary2 = glm::max(upperBoundary2, tri.v[i].dot(axisNormal));
            }
        }
        
        return false;//(lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2, intersectionDistance, boundary1, boundary2);
    }


    template<typename Real>
    DYN_FUNC inline bool checkOverlapAxis(
        Real& lowerBoundary1,
        Real& upperBoundary1,
        Real& lowerBoundary2,
        Real& upperBoundary2,
        Real& intersectionDistance,
        Real& boundary1,
        Real& boundary2,
        const Vector<Real, 3> axisNormal,
        Tet3D tet,
        OrientedBox3D box)
    {

        //projection to axis
        for (int i = 0; i < 4; i++)
        {
            if (i == 0)
                lowerBoundary1 = upperBoundary1 = tet.v[0].dot(axisNormal);
            else
            {
                lowerBoundary1 = glm::min(lowerBoundary1, tet.v[i].dot(axisNormal));
                upperBoundary1 = glm::max(upperBoundary1, tet.v[i].dot(axisNormal));
            }
        }

        Vector<Real, 3> center = box.center;
        Vector<Real, 3> u = box.u;
        Vector<Real, 3> v = box.v;
        Vector<Real, 3> w = box.w;
        Vector<Real, 3> extent = box.extent;
        Vector<Real, 3> p;
        p = (center - u * extent[0] - v * extent[1] - w * extent[2]);
        lowerBoundary2 = upperBoundary2 = p.dot(axisNormal);
        
        p = (center - u * extent[0] - v * extent[1] + w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center - u * extent[0] + v * extent[1] - w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));
        
        p = (center - u * extent[0] + v * extent[1] + w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));
        
        p = (center + u * extent[0] - v * extent[1] - w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center + u * extent[0] - v * extent[1] + w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center + u * extent[0] + v * extent[1] - w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));

        p = (center + u * extent[0] + v * extent[1] + w * extent[2]);
        lowerBoundary2 = glm::min(lowerBoundary2, p.dot(axisNormal));
        upperBoundary2 = glm::max(upperBoundary2, p.dot(axisNormal));
        
        return checkOverlap(lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2, intersectionDistance, boundary1, boundary2);
    }


    template<typename Real>
    DYN_FUNC inline void setupContactTets(
        Real boundary1,
        Real boundary2,
        const Vector<Real, 3> axisNormal,
        Tet3D tet1,
        Tet3D tet2,
        Real sMax,
        TManifold<Real>& m)
    {
        int cnt1, cnt2;
        unsigned char boundaryPoints1[4], boundaryPoints2[4];
        cnt1 = cnt2 = 0;

        

        for (unsigned char i = 0; i < 4; i++)
        {
            if (abs(tet1.v[i].dot(axisNormal) - boundary1) < abs(sMax))
                boundaryPoints1[cnt1 ++] = i;
            if (abs(tet2.v[i].dot(axisNormal) - boundary2) < abs(sMax))
                boundaryPoints2[cnt2 ++] = i;
        }
        //printf("cnt1 = %d, cnt2 = %d\n", cnt1, cnt2);
        if (cnt1 == 1 || cnt2 == 1)
        {
            
            m.normal = (boundary1 > boundary2) ? axisNormal : - axisNormal;
            m.contacts[0].penetration = sMax;
            m.contacts[0].position = (cnt1 == 1) ? tet1.v[boundaryPoints1[0]] : tet2.v[boundaryPoints2[0]];
            m.contactCount = 1;
            return;
        }
        else if (cnt1 == 2)
        {
            Segment3D s1(tet1.v[boundaryPoints1[0]], tet1.v[boundaryPoints1[1]]);
            
            if (cnt2 == 2)
            {
                Segment3D s2(tet2.v[boundaryPoints2[0]], tet2.v[boundaryPoints2[1]]);
                Segment3D dir = s1.proximity(s2);//v0: self v1: other
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                m.contacts[0].penetration = sMax;
                m.contacts[0].position = dir.v0;
                m.contactCount = 1;
                return;
            }
            else //cnt2 == 3
            {
                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                Triangle3D t2(tet2.v[boundaryPoints2[0]], tet2.v[boundaryPoints2[1]], tet2.v[boundaryPoints2[2]]);
                Vector<Real, 3> dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
                Vector<Real, 3> dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;
                
                if (dirTmp1.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s1.v0;
                    m.contactCount ++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s1.v1;
                    m.contactCount++;
                }
                
                for (int i = 0; i < 3; i++)
                {
                    Segment3D s2(t2.v[(i + 1) % 3], t2.v[(i + 2) % 3]);
                    Segment3D dir = s1.proximity(s2);
                    /*printf("dir: %.3lf %.3lf %.3lf\naxisnormal %.3lf %.3lf %.3lf\n%.6lf\n",
                        dir.direction()[0], dir.direction()[1], dir.direction()[2],
                        axisNormal[0], axisNormal[1], axisNormal[2],
                        dir.direction().normalize().cross(axisNormal).norm());*/
                    if ((!dir.isValid()) ||
                        (
                        (dir.direction().dot(s1.direction()) < 1e-5) && (dir.direction().dot(s2.direction()) < 1e-5)
                            ))
                    //if(dir.norm() < 1e5)
                    {
                        //printf("Yes\n");
                        if ((dir.v0 - s1.v0).norm() > 1e-5 && (dir.v0 - s1.v1).norm() > 1e-5)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                        }
                    }
                }
            }
        }
        else if (cnt1 == 3)
        {
            Triangle3D t1(tet1.v[boundaryPoints1[0]], tet1.v[boundaryPoints1[1]], tet1.v[boundaryPoints1[2]]);
            if (cnt2 == 2)
            {

                Segment3D s2(tet2.v[boundaryPoints2[0]], tet2.v[boundaryPoints2[1]]);
                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                
                Vector<Real, 3> dirTmp1 = Point3D(s2.v0).project(t1).origin - s2.v0;
                Vector<Real, 3> dirTmp2 = Point3D(s2.v1).project(t1).origin - s2.v1;
                if (dirTmp1.cross(axisNormal).norm() < 1e-5)

                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s2.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-5)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s2.v1;
                    m.contactCount++;
                }
                
                for (int i = 0; i < 3; i++)
                {
                    Segment3D s1(t1.v[(i + 1) % 3], t1.v[(i + 2) % 3]);
                    Segment3D dir = s2.proximity(s1);
                    if ((!dir.isValid()) ||
                        (
                        (dir.direction().dot(s1.direction()) < 1e-5) && (dir.direction().dot(s2.direction()) < 1e-5)
                        ))
                    {
                        if ((dir.v0 - s2.v0).norm() > 1e-5 && (dir.v0 - s2.v1).norm() > 1e-5)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                        }
                    }
                }
            }
            if (cnt2 == 3)
            {
                Triangle3D t1(tet1.v[boundaryPoints1[0]], tet1.v[boundaryPoints1[1]], tet1.v[boundaryPoints1[2]]);
                Triangle3D t2(tet2.v[boundaryPoints2[0]], tet2.v[boundaryPoints2[1]], tet2.v[boundaryPoints2[2]]);

                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;

                for (int i = 0; i < 3; i++)
                {
                    if ((Point3D(t1.v[i]).project(t2).origin - t1.v[i]).cross(t2.normal()).norm() < 1e-5)
                    {
                        m.contacts[m.contactCount].penetration = sMax;
                        m.contacts[m.contactCount].position = t1.v[i];
                        m.contactCount++;
                    }
                    if ((Point3D(t2.v[i]).project(t1).origin - t2.v[i]).cross(t1.normal()).norm() < 1e-5)
                    {
                        m.contacts[m.contactCount].penetration = sMax;
                        m.contacts[m.contactCount].position = t2.v[i];
                        m.contactCount++;
                    }
                    
                    for (int j = 0; j < 3; j++)
                    {
                        Segment3D s1(t1.v[(i + 1) % 3], t1.v[(i + 2) % 3]);
                        Segment3D s2(t2.v[(j + 1) % 3], t2.v[(j + 2) % 3]);
                        Segment3D dir = s1.proximity(s2);
                        

                        if ((!dir.isValid()) ||
                            (
                            (dir.direction().dot(s1.direction()) < 1e-5) && (dir.direction().dot(s2.direction()) < 1e-5)
                            )
                            )
                        {
                            if ((dir.v0 - s1.v0).norm() > 1e-5 && (dir.v0 - s1.v1).norm() > 1e-5)
                            {
                                m.contacts[m.contactCount].penetration = sMax;
                                m.contacts[m.contactCount].position = dir.v0;
                                m.contactCount++;
                            }
                        }
                    }
                    
                }

            }
        }
    }


    template<typename Real>
    DYN_FUNC inline void setupContactTets(
        Real boundary1,
        Real boundary2,
        const Vector<Real, 3> axisNormal,
        Tet3D tet,
        TOrientedBox3D<Real> box,
        Real sMax,
        TManifold<Real>& m)
    {
        int cnt1, cnt2;
        unsigned char boundaryPoints1[4];
        Vector<Real, 3> boundaryPoints2[8];
        cnt1 = cnt2 = 0;



        for (unsigned char i = 0; i < 4; i++)
        {
            if (abs(tet.v[i].dot(axisNormal) - boundary1) < abs(sMax))
                boundaryPoints1[cnt1++] = i;
        }

        Vector<Real, 3> center = box.center;
        Vector<Real, 3> u = box.u;
        Vector<Real, 3> v = box.v;
        Vector<Real, 3> w = box.w;
        Vector<Real, 3> extent = box.extent;
        Vector<Real, 3> p;
        p = (center - u * extent[0] - v * extent[1] - w * extent[2]);
        if (abs(p.dot(axisNormal) - boundary2) < abs(sMax))
            boundaryPoints2[cnt2++] = p;

        p = (center - u * extent[0] - v * extent[1] + w * extent[2]);
        if (abs(p.dot(axisNormal) - boundary2) < abs(sMax))
            boundaryPoints2[cnt2++] = p;

        p = (center - u * extent[0] + v * extent[1] - w * extent[2]);
        if (abs(p.dot(axisNormal) - boundary2) < abs(sMax))
            boundaryPoints2[cnt2++] = p;

        p = (center - u * extent[0] + v * extent[1] + w * extent[2]);
        if (abs(p.dot(axisNormal) - boundary2) < abs(sMax))
            boundaryPoints2[cnt2++] = p;

        p = (center + u * extent[0] - v * extent[1] - w * extent[2]);
        if (abs(p.dot(axisNormal) - boundary2) < abs(sMax))
            boundaryPoints2[cnt2++] = p;

        p = (center + u * extent[0] - v * extent[1] + w * extent[2]);
        if (abs(p.dot(axisNormal) - boundary2) < abs(sMax))
            boundaryPoints2[cnt2++] = p;

        p = (center + u * extent[0] + v * extent[1] - w * extent[2]);
        if (abs(p.dot(axisNormal) - boundary2) < abs(sMax))
            boundaryPoints2[cnt2++] = p;

        p = (center + u * extent[0] + v * extent[1] + w * extent[2]);
        if (abs(p.dot(axisNormal) - boundary2) < abs(sMax))
            boundaryPoints2[cnt2++] = p;

        //printf("cnt1 = %d, cnt2 = %d  %.3lf\n", cnt1, cnt2, sMax);
        if (cnt1 == 1 || cnt2 == 1)
        {
            m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
            m.contacts[0].penetration = sMax;
            m.contacts[0].position = (cnt1 == 1) ? tet.v[boundaryPoints1[0]] : boundaryPoints2[0];
            m.contactCount = 1;
            return;
        }
        else if (cnt1 == 2)
        {
            Segment3D s1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]]);

            if (cnt2 == 2)
            {
                Segment3D s2(boundaryPoints2[0], boundaryPoints2[1]);
                Segment3D dir = s1.proximity(s2);//v0: self v1: other
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                m.contacts[0].penetration = sMax;
                m.contacts[0].position = dir.v0;
                m.contactCount = 1;
                return;
            }
            else //cnt2 == 4
            {
                //if (cnt2 != 4)
                //	printf("?????????\n");

                
                for(int tmp_i = 1; tmp_i < 4; tmp_i ++)
                    for (int tmp_j = tmp_i + 1; tmp_j < 4; tmp_j++)
                        {
                            if ((boundaryPoints2[tmp_i] - boundaryPoints2[0]).dot(boundaryPoints2[tmp_j] - boundaryPoints2[0]) < EPSILON)
                            {
                                int tmp_k = 1 + 2 + 3 - tmp_i - tmp_j;
                                Vector<Real, 3> p2 = boundaryPoints2[tmp_i];
                                Vector<Real, 3> p3 = boundaryPoints2[tmp_j];
                                Vector<Real, 3> p4 = boundaryPoints2[tmp_k];
                                boundaryPoints2[1] = p2;
                                boundaryPoints2[2] = p3;
                                boundaryPoints2[3] = p4;
                                break;
                            }
                        }

                //printf("%.3lf %.3lf %.3lf\n", boundaryPoints2[0][0], boundaryPoints2[0][1], boundaryPoints2[0][2]);
                //printf("%.3lf %.3lf %.3lf\n", boundaryPoints2[1][0], boundaryPoints2[1][1], boundaryPoints2[1][2]);
                //printf("%.3lf %.3lf %.3lf\n", boundaryPoints2[2][0], boundaryPoints2[2][1], boundaryPoints2[2][2]);
                //printf("%.3lf %.3lf %.3lf\n", boundaryPoints2[3][0], boundaryPoints2[3][1], boundaryPoints2[3][2]);

                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;
                Triangle3D t2(boundaryPoints2[0], boundaryPoints2[1], boundaryPoints2[2]);
                Vector<Real, 3> dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
                Vector<Real, 3> dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;

                /*printf("%.3lf %.3lf %.3lf\n", axisNormal[0], axisNormal[1], axisNormal[2]);
                printf("%.3lf %.3lf %.3lf %.5lf %.5lf %.5lf\n", dirTmp1[0], dirTmp1[1], dirTmp1[2], dirTmp1.cross(axisNormal)[0], dirTmp1.cross(axisNormal)[1], dirTmp1.cross(axisNormal)[2]);
                printf("%.3lf %.3lf %.3lf %.5lf %.5lf %.5lf\n", dirTmp2[0], dirTmp2[1], dirTmp2[2], dirTmp2.cross(axisNormal)[0], dirTmp2.cross(axisNormal)[1], dirTmp2.cross(axisNormal)[2]);*/


                if (dirTmp1.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s1.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s1.v1;
                    m.contactCount++;
                }
                t2 = Triangle3D(boundaryPoints2[3], boundaryPoints2[1], boundaryPoints2[2]);
                dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
                dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;
                
                /*printf("%.3lf %.3lf %.3lf\n", dirTmp1[0], dirTmp1[1], dirTmp1[2]);
                printf("%.3lf %.3lf %.3lf\n", dirTmp2[0], dirTmp2[1], dirTmp2[2]);*/

                if (dirTmp1.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s1.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s1.v1;
                    m.contactCount++;
                }
                for (int i = 0; i < 4; i++)
                {
                    Segment3D s2;
                    if (i < 2)
                        s2 = Segment3D(boundaryPoints2[0], boundaryPoints2[i]);
                    else
                        s2 = Segment3D(boundaryPoints2[3], boundaryPoints2[i - 2]);
                    Segment3D dir = s1.proximity(s2);
                    /*printf("dir: %.3lf %.3lf %.3lf\naxisnormal %.3lf %.3lf %.3lf\n%.6lf\n",
                        dir.direction()[0], dir.direction()[1], dir.direction()[2],
                        axisNormal[0], axisNormal[1], axisNormal[2],
                        dir.direction().normalize().cross(axisNormal).norm());*/
                    if ((!dir.isValid()) || dir.direction().normalize().cross(axisNormal).norm() < 1e-4)
                    {
                        //printf("Yes\n");
                        if ((dir.v0 - s1.v0).norm() > 1e-4 && (dir.v0 - s1.v1).norm() > 1e-4)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                        }
                    }
                }
            }
        }
        else if (cnt1 == 3)
        {
            Triangle3D t1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]], tet.v[boundaryPoints1[2]]);

            //printf("%.3lf %.3lf %.3lf\n", axisNormal[0], axisNormal[1], axisNormal[2]);

            if (cnt2 == 2)
            {

                Segment3D s2(boundaryPoints2[0], boundaryPoints2[1]);
                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;

                Vector<Real, 3> dirTmp1 = Point3D(s2.v0).project(t1).origin - s2.v0;
                Vector<Real, 3> dirTmp2 = Point3D(s2.v1).project(t1).origin - s2.v1;
                if (dirTmp1.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s2.v0;
                    m.contactCount++;
                }
                if (dirTmp2.cross(axisNormal).norm() < 1e-4)
                {
                    m.contacts[m.contactCount].penetration = sMax;
                    m.contacts[m.contactCount].position = s2.v1;
                    m.contactCount++;
                }
                for (int i = 0; i < 3; i++)
                {
                    Segment3D s1(t1.v[(i + 1) % 3], t1.v[(i + 2) % 3]);
                    Segment3D dir = s2.proximity(s1);
                    if ((!dir.isValid()) || dir.direction().normalize().cross(axisNormal).norm() < 1e-4)
                    {
                        if ((dir.v0 - s2.v0).norm() > 1e-4 && (dir.v0 - s2.v1).norm() > 1e-4)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                        }
                    }
                }
            }
            else
            {
                Triangle3D t1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]], tet.v[boundaryPoints1[2]]);
                //Triangle3D t2(tet2.v[boundaryPoints2[0]], tet2.v[boundaryPoints2[1]], tet2.v[boundaryPoints2[2]]);

                //if (cnt2 != 4)
                //	printf("?????????\n");


                for (int tmp_i = 1; tmp_i < 4; tmp_i++)
                    for (int tmp_j = tmp_i + 1; tmp_j < 4; tmp_j++)
                    {
                        if ((boundaryPoints2[tmp_i] - boundaryPoints2[0]).dot(boundaryPoints2[tmp_j] - boundaryPoints2[0]) < EPSILON)
                        {
                            int tmp_k = 1 + 2 + 3 - tmp_i - tmp_j;
                            Vector<Real, 3> p2 = boundaryPoints2[tmp_i];
                            Vector<Real, 3> p3 = boundaryPoints2[tmp_j];
                            Vector<Real, 3> p4 = boundaryPoints2[tmp_k];
                            boundaryPoints2[1] = p2;
                            boundaryPoints2[2] = p3;
                            boundaryPoints2[3] = p4;
                            break;
                        }
                    }

                m.contactCount = 0;
                m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;

                for(int i = 0; i < 4; i ++)
                    if ((Point3D(boundaryPoints2[i]).project(t1).origin - boundaryPoints2[i]).cross(t1.normal()).norm() < 1e-4)
                    {
                        m.contacts[m.contactCount].penetration = sMax;
                        m.contacts[m.contactCount].position = boundaryPoints2[i];
                        m.contactCount++;
                        //printf("b");
                    }

                for (int i = 0; i < 3; i++)
                {
                    Triangle3D t2(boundaryPoints2[0], boundaryPoints2[1], boundaryPoints2[2]);
                    if ((Point3D(t1.v[i]).project(t2).origin - t1.v[i]).cross(t2.normal()).norm() < 1e-4)
                    {
                        m.contacts[m.contactCount].penetration = sMax;
                        m.contacts[m.contactCount].position = t1.v[i];
                        m.contactCount++;
                        //printf("a1");
                    }
                    t2 = Triangle3D(boundaryPoints2[3], boundaryPoints2[1], boundaryPoints2[2]);
                    if ((Point3D(t1.v[i]).project(t2).origin - t1.v[i]).cross(t2.normal()).norm() < 1e-4)
                    {
                        m.contacts[m.contactCount].penetration = sMax;
                        m.contacts[m.contactCount].position = t1.v[i];
                        m.contactCount++;
                        //printf("a2");
                    }
                    

                    Segment3D s1(t1.v[(i + 1) % 3], t1.v[(i + 2) % 3]);
                    Segment3D s2(boundaryPoints2[0], boundaryPoints2[1]);
                    Segment3D dir = s1.proximity(s2);
                    if ((!dir.isValid()) || dir.direction().cross(axisNormal).norm() < 1e-4)
                    {
                        if ((dir.v0 - s1.v0).norm() > 1e-4 && (dir.v0 - s1.v1).norm() > 1e-4)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                            //printf("c");
                        }

                    }

                    s2 = Segment3D(boundaryPoints2[0], boundaryPoints2[2]);
                    dir = s1.proximity(s2);
                    if ((!dir.isValid()) || dir.direction().cross(axisNormal).norm() < 1e-4)
                    {
                        if ((dir.v0 - s1.v0).norm() > 1e-4 && (dir.v0 - s1.v1).norm() > 1e-4)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                            //printf("c");
                        }
                    }
                    s2 = Segment3D(boundaryPoints2[3], boundaryPoints2[2]);
                    dir = s1.proximity(s2);
                    if ((!dir.isValid()) || dir.direction().cross(axisNormal).norm() < 1e-4)
                    {
                        if ((dir.v0 - s1.v0).norm() > 1e-4 && (dir.v0 - s1.v1).norm() > 1e-4)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                            //printf("c");
                        }
                    }
                    s2 = Segment3D(boundaryPoints2[3], boundaryPoints2[1]);
                    dir = s1.proximity(s2);
                    if ((!dir.isValid()) || dir.direction().cross(axisNormal).norm() < 1e-4)
                    {
                        if ((dir.v0 - s1.v0).norm() > 1e-4 && (dir.v0 - s1.v1).norm() > 1e-4)
                        {
                            m.contacts[m.contactCount].penetration = sMax;
                            m.contacts[m.contactCount].position = dir.v0;
                            m.contactCount++;
                            //printf("c");
                        }
                    }
                    
                }

            }
        }
    }

    //Separating Axis Theorem for tets
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tet0, const Tet3D& tet1)
    {
        m.contactCount = 0;

        Real sMax = (Real)INT_MAX;
        Real sIntersect;
        Real lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2;
        Real l1, u1, l2, u2;
        Vector<Real, 3> axis = Vector<Real, 3>(0, 1, 0);
        Vector<Real, 3> axisTmp = axis;

        Real boundary1, boundary2, b1, b2;


        // no penetration when the tets are illegal
        if (abs(tet0.volume()) < EPSILON || abs(tet1.volume()) < EPSILON)
            return;
        
        for(int i = 0; i < 4; i ++)
        { 
            //tet0 face axis i
            axisTmp = tet0.face(i).normal();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet0, tet1) == false)
            {
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }
            //tet1 face axis i
            axisTmp = tet1.face(i).normal();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet0, tet1) == false)
            { 
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }
        }

        const int segmentIndex[6][2] = {
        0, 1,
        0, 2,
        0, 3,
        1, 2,
        1, 3,
        2, 3
        };

        for(int i = 0; i < 6; i ++)
            for (int j = 0; j < 6; j++)
            {
                Vector<Real, 3> dirTet1 = tet0.v[segmentIndex[i][0]] - tet0.v[segmentIndex[i][1]];
                Vector<Real, 3> dirTet2 = tet1.v[segmentIndex[j][0]] - tet1.v[segmentIndex[j][1]];
                axisTmp = dirTet1.cross(dirTet2);
                if (axisTmp.norm() > EPSILON)
                {
                    axisTmp /= axisTmp.norm();
                }
                else //parallel, choose an arbitary direction
                {
                    if (abs(dirTet1[0]) > EPSILON)
                        axisTmp = Vector<Real, 3>(dirTet1[1], -dirTet1[0], 0);
                    else
                        axisTmp = Vector<Real, 3>(0, dirTet1[2], -dirTet1[1]);
                    axisTmp /= axisTmp.norm();
                }
                if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet0, tet1) == false)
                { 
                    m.contactCount = 0;
                    return;
                }
                else
                {
                    if (sIntersect < sMax)
                    {
                        sMax = sIntersect;
                        lowerBoundary1 = l1;
                        lowerBoundary2 = l2;
                        upperBoundary1 = u1;
                        upperBoundary2 = u2;
                        boundary1 = b1;
                        boundary2 = b2;
                        axis = axisTmp;
                    }
                }
            }
        //printf("YES YYYYYYYEEES\n!\n");
        //set up contacts using axis
        setupContactTets(boundary1, boundary2, axis, tet0, tet1, -sMax, m);
    }


    //Separating Axis Theorem for tet-OBB
    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tet, const OBox3D& box)
    {
        m.contactCount = 0;

        Real sMax = (Real)INT_MAX;
        Real sIntersect;
        Real lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2;
        Real l1, u1, l2, u2;
        Vector<Real, 3> axis = Vector<Real, 3>(0, 1, 0);
        Vector<Real, 3> axisTmp = axis;

        Real boundary1, boundary2, b1, b2;

        for (int i = 0; i < 4; i++)
        {
            //tet face axis i
            axisTmp = tet.face(i).normal();
            if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, box) == false)
            {
                m.contactCount = 0;
                return;
            }
            else
            {
                if (sIntersect < sMax)
                {
                    sMax = sIntersect;
                    lowerBoundary1 = l1;
                    lowerBoundary2 = l2;
                    upperBoundary1 = u1;
                    upperBoundary2 = u2;
                    boundary1 = b1;
                    boundary2 = b2;
                    axis = axisTmp;
                }
            }
        }

        //u
        axisTmp = box.u / box.u.norm();
        if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, box) == false)
        {
            m.contactCount = 0;
            return;
        }
        else
        {
            if (sIntersect < sMax)
            {
                sMax = sIntersect;
                lowerBoundary1 = l1;
                lowerBoundary2 = l2;
                upperBoundary1 = u1;
                upperBoundary2 = u2;
                boundary1 = b1;
                boundary2 = b2;
                axis = axisTmp;
            }
        }


        //v
        axisTmp = box.v / box.v.norm();
        if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, box) == false)
        {
            m.contactCount = 0;
            return;
        }
        else
        {
            if (sIntersect < sMax)
            {
                sMax = sIntersect;
                lowerBoundary1 = l1;
                lowerBoundary2 = l2;
                upperBoundary1 = u1;
                upperBoundary2 = u2;
                boundary1 = b1;
                boundary2 = b2;
                axis = axisTmp;
            }
        }

        //w
        axisTmp = box.w / box.w.norm();
        if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, box) == false)
        {
            m.contactCount = 0;
            return;
        }
        else
        {
            if (sIntersect < sMax)
            {
                sMax = sIntersect;
                lowerBoundary1 = l1;
                lowerBoundary2 = l2;
                upperBoundary1 = u1;
                upperBoundary2 = u2;
                boundary1 = b1;
                boundary2 = b2;
                axis = axisTmp;
            }
        }

        const int segmentIndex[6][2] = {
        0, 1,
        0, 2,
        0, 3,
        1, 2,
        1, 3,
        2, 3
        };

        //dir generated by cross product from tet and box
        for (int i = 0; i < 6; i++)
        {
            Vector<Real, 3> dirTet = tet.v[segmentIndex[i][0]] - tet.v[segmentIndex[i][1]];
            for(int j = 0; j < 3; j ++)
            { 
                Vector<Real, 3> boxDir = (j == 0) ? (box.u) : ((j == 1) ? (box.v) : (box.w));
                axisTmp = dirTet.cross(boxDir);
                if (axisTmp.norm() > EPSILON)
                {
                    axisTmp /= axisTmp.norm();
                }
                else //parallel, choose an arbitary direction
                {
                    if (abs(dirTet[0]) > EPSILON)
                        axisTmp = Vector<Real, 3>(dirTet[1], -dirTet[0], 0);
                    else
                        axisTmp = Vector<Real, 3>(0, dirTet[2], -dirTet[1]);
                    axisTmp /= axisTmp.norm();
                }
                if (checkOverlapAxis(l1, u1, l2, u2, sIntersect, b1, b2, axisTmp, tet, box) == false)
                {
                    m.contactCount = 0;
                    return;
                }
                else
                {
                    if (sIntersect < sMax)
                    {
                        sMax = sIntersect;
                        lowerBoundary1 = l1;
                        lowerBoundary2 = l2;
                        upperBoundary1 = u1;
                        upperBoundary2 = u2;
                        boundary1 = b1;
                        boundary2 = b2;
                        axis = axisTmp;
                    }
                }
            }
        }
        setupContactTets(boundary1, boundary2, axis, tet, box, -sMax, m);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& box, const Tet3D& tet)
    {
        request(m, tet, box);

        swapContactPair(m);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphere, const Tet3D& tet)
    {
        m.contactCount = 0;

        Point3D c0(sphere.center);
        Real r0 = sphere.radius;

        Point3D vproj = c0.project(tet);
        bool bInside = c0.inside(tet);

        Segment3D dir = bInside ? c0 - vproj : vproj - c0;
        Real sMax = bInside ? -dir.direction().norm() - r0 : dir.direction().norm() - r0;

        if (sMax >= 0)
            return;

        m.normal = dir.direction().normalize();
        m.contacts[0].penetration = sMax;
        m.contacts[0].position = vproj.origin;
        m.contactCount = 1;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tet, const Sphere3D& sphere)
    {
        request(m, sphere, tet);

        swapContactPair(m);
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphere, const Triangle3D& tri)
    {
        m.contactCount = 0;

        Point3D c0(sphere.center);
        Real r0 = sphere.radius;

        Point3D vproj = c0.project(tri);
        

        Segment3D dir = vproj - c0;
        Real sMax = dir.direction().norm() - r0;

        if (sMax >= 0)
            return;

        /*if (dir.direction().norm() < EPSILON)
            return;*/

        /*if ((dir.direction().normalize().cross(tri.normal().normalize())).norm() > EPSILON)
            return;*/

        m.normal = dir.direction().normalize();
        m.contacts[0].penetration = sMax;
        m.contacts[0].position = vproj.origin;
        m.contactCount = 1;
    }

    template<typename Real>
    DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Triangle3D& tri, const Sphere3D& sphere)
    {
        request(m, sphere, tri);
        m.normal *= -1;
    }
}