#include "Topology/Primitive3D.h"

namespace dyno
{

	

	struct ClipVertex
	{
		Vector<Real, 3> v;
	};

	template<typename Real>
	DYN_FUNC float fsign(Real v)
	{
		return v < 0 ? -Real(1) : Real(1);
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
	// http://www.randygaul.net/2013/10/27/sutherland-hodgman-clipping/
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

		aOut = rot * trans + a;
		bOut = rot * trans + b;
	}

	template<typename Real>
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D box0, const OBox3D box1)
	{
		m.contactCount = 0;

		Coord3D v = box1.center - box0.center;

		Coord3D eA = box0.extent;
		Coord3D eB = box1.extent;

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
		Coord3D t = rotA.transpose() * v;

		// Query states
		Real s;
		Real aMax = -REAL_MAX;
		Real bMax = -REAL_MAX;
		float eMax = -REAL_MAX;
		int aAxis = ~0;
		int bAxis = ~0;
		int eAxis = ~0;
		Coord3D nA;
		Coord3D nB;
		Coord3D nE;

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
			if (trackEdgeAxis(eAxis, eMax, nE, 6, s, Coord3D(float(0.0), -C(2, 0), C(1, 0))))
				return;

			// Cross( a.x, b.y )
			rA = eA.y * absC(2, 1) + eA.z * absC(1, 1);
			rB = eB.x * absC(0, 2) + eB.z * absC(0, 0);
			s = abs(t.z * C(1, 1) - t.y * C(2, 1)) - (rA + rB);
			if (trackEdgeAxis(eAxis, eMax, nE, 7, s, Coord3D(float(0.0), -C(2, 1), C(1, 1))))
				return;

			// Cross( a.x, b.z )
			rA = eA.y * absC(2, 2) + eA.z * absC(1, 2);
			rB = eB.x * absC(0, 1) + eB.y * absC(0, 0);
			s = abs(t.z * C(1, 2) - t.y * C(2, 2)) - (rA + rB);
			if (trackEdgeAxis(eAxis, eMax, nE, 8, s, Coord3D(float(0.0), -C(2, 2), C(1, 2))))
				return;

			// Cross( a.y, b.x )
			rA = eA.x * absC(2, 0) + eA.z * absC(0, 0);
			rB = eB.y * absC(1, 2) + eB.z * absC(1, 1);
			s = abs(t.x * C(2, 0) - t.z * C(0, 0)) - (rA + rB);
			if (trackEdgeAxis(eAxis, eMax, nE, 9, s, Coord3D(C(2, 0), float(0.0), -C(0, 0))))
				return;

			// Cross( a.y, b.y )
			rA = eA.x * absC(2, 1) + eA.z * absC(0, 1);
			rB = eB.x * absC(1, 2) + eB.z * absC(1, 0);
			s = abs(t.x * C(2, 1) - t.z * C(0, 1)) - (rA + rB);
			if (trackEdgeAxis(eAxis, eMax, nE, 10, s, Coord3D(C(2, 1), float(0.0), -C(0, 1))))
				return;

			// Cross( a.y, b.z )
			rA = eA.x * absC(2, 2) + eA.z * absC(0, 2);
			rB = eB.x * absC(1, 1) + eB.y * absC(1, 0);
			s = abs(t.x * C(2, 2) - t.z * C(0, 2)) - (rA + rB);
			if (trackEdgeAxis(eAxis, eMax, nE, 11, s, Coord3D(C(2, 2), float(0.0), -C(0, 2))))
				return;

			// Cross( a.z, b.x )
			rA = eA.x * absC(1, 0) + eA.y * absC(0, 0);
			rB = eB.y * absC(2, 2) + eB.z * absC(2, 1);
			s = abs(t.y * C(0, 0) - t.x * C(1, 0)) - (rA + rB);
			if (trackEdgeAxis(eAxis, eMax, nE, 12, s, Coord3D(-C(1, 0), C(0, 0), float(0.0))))
				return;

			// Cross( a.z, b.y )
			rA = eA.x * absC(1, 1) + eA.y * absC(0, 1);
			rB = eB.x * absC(2, 2) + eB.z * absC(2, 0);
			s = abs(t.y * C(0, 1) - t.x * C(1, 1)) - (rA + rB);
			if (trackEdgeAxis(eAxis, eMax, nE, 13, s, Coord3D(-C(1, 1), C(0, 1), float(0.0))))
				return;

			// Cross( a.z, b.z )
			rA = eA.x * absC(1, 2) + eA.y * absC(0, 2);
			rB = eB.x * absC(2, 1) + eB.y * absC(2, 0);
			s = abs(t.y * C(0, 2) - t.x * C(1, 2)) - (rA + rB);
			if (trackEdgeAxis(eAxis, eMax, nE, 14, s, Coord3D(-C(1, 2), C(0, 2), float(0.0))))
				return;
		}

		// Artificial axis bias to improve frame coherence
		const float kRelTol = float(0.95);
		const float kAbsTol = float(0.01);
		int axis;
		float sMax;
		Coord3D n;
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
			Coord3D eR;
			Coord3D eI;
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
			Coord3D e;
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
					m.contacts[i].position = out[i].v;
					m.contacts[i].penetration = depths[i];
				}
			}
		}
		else
		{
			n = rotA * n;

			if (n.dot(v) < float(0.0))
				n = -n;

			Coord3D PA, QA;
			Coord3D PB, QB;
			computeSupportEdge(PA, QA, rotA, box0.center, eA, n);
			computeSupportEdge(PB, QB, rotB, box1.center, eB, -n);

			Coord3D CA, CB;
			edgesContact(CA, CB, PA, QA, PB, QB);

			m.normal = n;
			m.contactCount = 1;

			m.contacts[0].penetration = sMax;
			m.contacts[0].position = (CA + CB) * float(0.5);
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

		Segment3D s0(cap0.segment);
		Segment3D s1(cap1.segment);
		Real r0 = cap0.radius + cap1.radius;

		Segment3D dir = s0.proximity(s1);

		dir = Point3D(dir.endPoint()) - Point3D(dir.startPoint());

		Real sMax = dir.direction().norm() - r0;
		if (sMax >= 0)
			return;

		m.normal = dir.direction().normalize();
		m.contacts[0].penetration = sMax;
		m.contacts[0].position = dir.v0;
		m.contactCount = 1;
	}


	template<typename Real>
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Sphere3D& sphere, const Capsule3D& cap)
	{
		m.contactCount = 0;

		Point3D s(sphere.center);
		Segment3D c(cap.segment);
		Real r0 = cap.radius + sphere.radius;

		Point3D pos = s.project(c);//pos: capsule

		Segment3D dir = pos - s;

		Real sMax = dir.direction().norm() - r0;
		if (sMax >= 0)
			return;

		m.normal = dir.direction().normalize();
		m.contacts[0].penetration = sMax;
		m.contacts[0].position = dir.v0;
		m.contactCount = 1;
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
		Segment3D seg = cap.segment;

		lowerBoundary1 = seg.v0.dot(axisNormal) - cap.radius;
		upperBoundary1 = seg.v0.dot(axisNormal) + cap.radius;
		lowerBoundary1 = glm::min(lowerBoundary1, seg.v1.dot(axisNormal) - cap.radius);
		upperBoundary1 = glm::max(upperBoundary1, seg.v1.dot(axisNormal) + cap.radius);
			
		

		Coord3D center = box.center;
		Coord3D u = box.u;
		Coord3D v = box.v;
		Coord3D w = box.w;
		Coord3D extent = box.extent;
		Coord3D p;
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

		Segment3D seg = cap.segment;
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
	DYN_FUNC inline void setupContactTets(
		Real boundary1,
		Real boundary2,
		const Vector<Real, 3> axisNormal,
		TCapsule3D<Real> cap,
		TOrientedBox3D<Real> box,
		Real sMax,
		TManifold<Real>& m)
	{
		int cnt1, cnt2;
		Coord3D boundaryPoints1[4];
		Coord3D boundaryPoints2[8];
		cnt1 = cnt2 = 0;

		if (abs(cap.segment.v0.dot(axisNormal) + cap.radius - boundary1) < abs(sMax)
			||
			abs(cap.segment.v0.dot(axisNormal) - cap.radius - boundary1) < abs(sMax))
			boundaryPoints2[cnt1++] = cap.segment.v0;
		if (abs(cap.segment.v1.dot(axisNormal) + cap.radius - boundary1) < abs(sMax)
			||
			abs(cap.segment.v1.dot(axisNormal) - cap.radius - boundary1) < abs(sMax))
			boundaryPoints2[cnt1++] = cap.segment.v1;

		

		Coord3D center = box.center;
		Coord3D u = box.u;
		Coord3D v = box.v;
		Coord3D w = box.w;
		Coord3D extent = box.extent;
		Coord3D p;
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
			m.contacts[0].position = (cnt1 == 1) ? boundaryPoints1[0] : boundaryPoints2[0];
			m.contactCount = 1;
			return;
		}
		else if (cnt1 == 2)
		{
			Segment3D s1 = cap.segment;

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


				for (int tmp_i = 1; tmp_i < 4; tmp_i++)
					for (int tmp_j = tmp_i + 1; tmp_j < 4; tmp_j++)
					{
						if ((boundaryPoints2[tmp_i] - boundaryPoints2[0]).dot(boundaryPoints2[tmp_j] - boundaryPoints2[0]) < EPSILON)
						{
							int tmp_k = 1 + 2 + 3 - tmp_i - tmp_j;
							Coord3D p2 = boundaryPoints2[tmp_i];
							Coord3D p3 = boundaryPoints2[tmp_j];
							Coord3D p4 = boundaryPoints2[tmp_k];
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
				Coord3D dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
				Coord3D dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;

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

		if (abs(cap.segment.v0.dot(axisNormal) + cap.radius - boundary1) < abs(sMax)
			||
			abs(cap.segment.v0.dot(axisNormal) - cap.radius - boundary1) < abs(sMax))
			boundaryPoints2[cnt2 ++] = 0;
		
		if (abs(cap.segment.v1.dot(axisNormal) + cap.radius - boundary1) < abs(sMax)
			||
			abs(cap.segment.v1.dot(axisNormal) - cap.radius - boundary1) < abs(sMax))
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
			m.contacts[0].position = boundaryPoints2[0] ? cap.segment.v1 : cap.segment.v0;
			m.contactCount = 1;
			return;
		}
		else if (cnt1 == 2)
		{
			Segment3D s1(tet.v[boundaryPoints1[0]], tet.v[boundaryPoints1[1]]);

			//if (cnt2 == 2)
			{
				Segment3D s2 = cap.segment;
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

				Segment3D s2 = cap.segment;
				m.contactCount = 0;
				m.normal = (boundary1 > boundary2) ? axisNormal : -axisNormal;

				Coord3D dirTmp1 = Point3D(s2.v0).project(t1).origin - s2.v0;
				Coord3D dirTmp2 = Point3D(s2.v1).project(t1).origin - s2.v1;
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
				Coord3D dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
				Coord3D dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;
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

				Coord3D dirTmp1 = Point3D(s2.v0).project(t1).origin - s2.v0;
				Coord3D dirTmp2 = Point3D(s2.v1).project(t1).origin - s2.v1;
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
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& box, const Capsule3D& cap)
	{
		Segment3D c(cap.segment);
		Real r0 = cap.radius;

		m.contactCount = 0;

		Real sMax = (Real)INT_MAX;
		Real sIntersect;
		Real lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2;
		Real l1, u1, l2, u2;
		Coord3D axis = Coord3D(0, 1, 0);
		Coord3D axisTmp = axis;

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
		
		Coord3D dirCap = c.direction();
		for (int j = 0; j < 3; j++)
		{
			Coord3D boxDir = (j == 0) ? (box.u) : ((j == 1) ? (box.v) : (box.w));
			axisTmp = dirCap.cross(boxDir);
			if (axisTmp.norm() > EPSILON)
			{
				axisTmp /= axisTmp.norm();
			}
			else //parallel, choose an arbitary direction
			{
				if (abs(dirCap[0]) > EPSILON)
					axisTmp = Coord3D(dirCap[1], -dirCap[0], 0);
				else
					axisTmp = Coord3D(0, dirCap[2], -dirCap[1]);
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
		
		setupContactTets(boundary1, boundary2, axis, cap, box, -sMax, m);

	}

	template<typename Real>
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tet, const Triangle3D& tri)
	{
		m.contactCount = 0;

		Real sMax = (Real)INT_MAX;
		Real sIntersect;
		Real lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2;
		Real l1, u1, l2, u2;
		Coord3D axis = Coord3D(0, 1, 0);
		Coord3D axisTmp = axis;

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
				Coord3D dirTet = tet.v[segmentIndex[i][0]] - tet.v[segmentIndex[i][1]];
				Coord3D dirTri = tri.v[triIndex[j][0]] - tri.v[triIndex[j][1]];
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
						axisTmp = Coord3D(dirTet[1], -dirTet[0], 0);
					else
						axisTmp = Coord3D(0, dirTet[2], -dirTet[1]);
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

	template<typename Real>
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Capsule3D& cap, const Tet3D& tet)
	{
		request(m, tet, cap);
		m.normal = -m.normal;
	}

	//Separating Axis Theorem for tets
	template<typename Real>
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tet, const Capsule3D& cap)
	{
		m.contactCount = 0;

		Real sMax = (Real)INT_MAX;
		Real sIntersect;
		Real lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2;
		Real l1, u1, l2, u2;
		Coord3D axis = Coord3D(0, 1, 0);
		Coord3D axisTmp = axis;

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

		axisTmp = cap.segment.direction();
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
			Coord3D dirTet = tet.v[segmentIndex[i][0]] - tet.v[segmentIndex[i][1]];
			Coord3D dirCap = cap.segment.direction();

			

			axisTmp = dirTet.cross(dirCap);
			if (axisTmp.norm() > EPSILON)
			{
				axisTmp /= axisTmp.norm();
			}
			else //parallel, choose an arbitary direction
			{
				if (abs(dirTet[0]) > EPSILON)
					axisTmp = Coord3D(dirTet[1], -dirTet[0], 0);
				else
					axisTmp = Coord3D(0, dirTet[2], -dirTet[1]);
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
		m.normal *= -1;
	}

	template<typename Real>
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const OBox3D& box, const Sphere3D& sphere)
	{
		request(m, sphere, box);
		m.normal = -m.normal;
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
		m.contacts[0].position = c0 + (r0 - 0.5 * sMax) * m.normal;
		m.contactCount = 1;
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
	DYN_FUNC inline bool checkOverlap(
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
					intersectionDistance = upperBoundary2 - lowerBoundary2;
					if (upperBoundary2 - lowerBoundary1 > upperBoundary1 - lowerBoundary2)
					{
						boundary1 = upperBoundary1;
						boundary2 = lowerBoundary2;
					}
					else
					{
						boundary1 = lowerBoundary1;
						boundary2 = upperBoundary2;
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
					intersectionDistance = upperBoundary1 - lowerBoundary1;
					if (upperBoundary2 - lowerBoundary1 > upperBoundary1 - lowerBoundary2)
					{
						boundary1 = upperBoundary1;
						boundary2 = lowerBoundary2;
					}
					else
					{
						boundary1 = lowerBoundary1;
						boundary2 = upperBoundary2;
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
		
		return checkOverlapTetTri(lowerBoundary1, upperBoundary1, lowerBoundary2, upperBoundary2, intersectionDistance, boundary1, boundary2);
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

		Coord3D center = box.center;
		Coord3D u = box.u;
		Coord3D v = box.v;
		Coord3D w = box.w;
		Coord3D extent = box.extent;
		Coord3D p;
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
				Coord3D dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
				Coord3D dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;
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
					if ( (!dir.isValid()) || dir.direction().normalize().cross(axisNormal).norm() < 1e-5)
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
				
				Coord3D dirTmp1 = Point3D(s2.v0).project(t1).origin - s2.v0;
				Coord3D dirTmp2 = Point3D(s2.v1).project(t1).origin - s2.v1;
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
		Coord3D boundaryPoints2[8];
		cnt1 = cnt2 = 0;



		for (unsigned char i = 0; i < 4; i++)
		{
			if (abs(tet.v[i].dot(axisNormal) - boundary1) < abs(sMax))
				boundaryPoints1[cnt1++] = i;
		}

		Coord3D center = box.center;
		Coord3D u = box.u;
		Coord3D v = box.v;
		Coord3D w = box.w;
		Coord3D extent = box.extent;
		Coord3D p;
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
								Coord3D p2 = boundaryPoints2[tmp_i];
								Coord3D p3 = boundaryPoints2[tmp_j];
								Coord3D p4 = boundaryPoints2[tmp_k];
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
				Coord3D dirTmp1 = Point3D(s1.v0).project(t2).origin - s1.v0;
				Coord3D dirTmp2 = Point3D(s1.v1).project(t2).origin - s1.v1;

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

				Coord3D dirTmp1 = Point3D(s2.v0).project(t1).origin - s2.v0;
				Coord3D dirTmp2 = Point3D(s2.v1).project(t1).origin - s2.v1;
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
							Coord3D p2 = boundaryPoints2[tmp_i];
							Coord3D p3 = boundaryPoints2[tmp_j];
							Coord3D p4 = boundaryPoints2[tmp_k];
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
		Coord3D axis = Coord3D(0, 1, 0);
		Coord3D axisTmp = axis;

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
				Coord3D dirTet1 = tet0.v[segmentIndex[i][0]] - tet0.v[segmentIndex[i][1]];
				Coord3D dirTet2 = tet1.v[segmentIndex[j][0]] - tet1.v[segmentIndex[j][1]];
				axisTmp = dirTet1.cross(dirTet2);
				if (axisTmp.norm() > EPSILON)
				{
					axisTmp /= axisTmp.norm();
				}
				else //parallel, choose an arbitary direction
				{
					if (abs(dirTet1[0]) > EPSILON)
						axisTmp = Coord3D(dirTet1[1], -dirTet1[0], 0);
					else
						axisTmp = Coord3D(0, dirTet1[2], -dirTet1[1]);
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
		Coord3D axis = Coord3D(0, 1, 0);
		Coord3D axisTmp = axis;

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
			Coord3D dirTet = tet.v[segmentIndex[i][0]] - tet.v[segmentIndex[i][1]];
			for(int j = 0; j < 3; j ++)
			{ 
				Coord3D boxDir = (j == 0) ? (box.u) : ((j == 1) ? (box.v) : (box.w));
				axisTmp = dirTet.cross(boxDir);
				if (axisTmp.norm() > EPSILON)
				{
					axisTmp /= axisTmp.norm();
				}
				else //parallel, choose an arbitary direction
				{
					if (abs(dirTet[0]) > EPSILON)
						axisTmp = Coord3D(dirTet[1], -dirTet[0], 0);
					else
						axisTmp = Coord3D(0, dirTet[2], -dirTet[1]);
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
		m.normal *= -1;
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
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Tet3D& tet, const Sphere3D& sphere)
	{
		request(m, sphere, tet);
		m.normal *= -1;
	}
	template<typename Real>
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, const Triangle3D& tri, const Sphere3D& sphere)
	{
		request(m, sphere, tri);
		m.normal *= -1;
	}
}