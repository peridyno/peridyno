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
	DYN_FUNC void CollisionDetection<Real>::request(Manifold& m, OBox3D box0, OBox3D box1)
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

		Transform3D atx(rotA, box0.center);
		Transform3D btx(rotB, box1.center);

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
}