//#include "Primitive3D.h"
#include "Algorithm/SimpleMath.h"
#include "Interval.h"
#include <glm/glm.hpp>

namespace dyno
{
	template<typename Real>
	DYN_FUNC TPoint3D<Real>::TPoint3D()
	{
		origin = Coord3D(0);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real>::TPoint3D(const Real& val)
	{
		origin = Coord3D(val);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real>::TPoint3D(const Real& c0, const Real& c1, const Real& c2)
	{
		origin = Coord3D(c0, c1, c2);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real>::TPoint3D(const Coord3D& pos)
	{
		origin = pos;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real>::TPoint3D(const TPoint3D& pt)
	{
		origin = pt.origin;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::operator=(const Coord3D& p)
	{
		TPoint3D<Real> pt;
		pt.origin = p;
		return pt;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TLine3D<Real>& line) const
	{
		Coord3D u = origin - line.origin;
		Real tNum = u.dot(line.direction);
		Real a = line.direction.normSquared();
		Real t = a < REAL_EPSILON_SQUARED ? 0 : tNum / a;

		return TPoint3D(line.origin + t * line.direction);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TRay3D<Real>& ray) const
	{
		Coord3D u = origin - ray.origin;

		Real tNum = u.dot(ray.direction);
		Real a = ray.direction.normSquared();
		Real t = a < REAL_EPSILON_SQUARED ? 0 : tNum / a;

		t = t < 0 ? 0 : t;

		return TPoint3D<Real>(ray.origin + t * ray.direction);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TSegment3D<Real>& segment) const
	{
		Coord3D l = origin - segment.v0;
		Coord3D dir = segment.v1 - segment.v0;
		if (dir.normSquared() < REAL_EPSILON_SQUARED)
		{
			return TPoint3D<Real>(segment.v0);
		}

		Real t = l.dot(dir) / dir.normSquared();

		Coord3D q = segment.v0 + t * dir;
		q = t < 0 ? segment.v0 : q;
		q = t > 1 ? segment.v1 : q;
		//printf("T: %.3lf\n", t);
		return TPoint3D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TPlane3D<Real>& plane) const
	{
		Real t = (origin - plane.origin).dot(plane.normal);

		Real n2 = plane.normal.normSquared();

		return n2 < REAL_EPSILON ? TPoint3D<Real>(plane.origin) : TPoint3D<Real>(origin - t / n2 * plane.normal);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TTriangle3D<Real>& triangle) const
	{
		Coord3D dir = triangle.v[0] - origin;
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Real a = e0.dot(e0);
		Real b = e0.dot(e1);
		Real c = e1.dot(e1);
		Real d = e0.dot(dir);
		Real e = e1.dot(dir);
		Real f = dir.dot(dir);

		Real det = a * c - b * b;
		Real s = b * e - c * d;
		Real t = b * d - a * e;

		Real maxL = triangle.maximumEdgeLength();
		//handle degenerate triangles
		if (det < REAL_EPSILON * maxL * maxL)
		{
			Real g = (triangle.v[2] - triangle.v[1]).normSquared();

			Real l_max = a;
			Coord3D p0 = triangle.v[0];
			Coord3D p1 = triangle.v[1];

			if (c > l_max)
			{
				p0 = triangle.v[0];
				p1 = triangle.v[2];

				l_max = c;
			}

			if (g > l_max)
			{
				p0 = triangle.v[1];
				p1 = triangle.v[2];
			}

			return project(TSegment3D<Real>(p0, p1));
		}

		if (s + t <= det)
		{
			if (s < 0)
			{
				if (t < 0)
				{
					//region 4
					s = 0;
					t = 0;
				}
				else
				{
					// region 3
					// F(t) = Q(0, t) = ct^2 + 2et + f
					// F'(t)/2 = ct + e
					// F'(t) = 0 when t = -e/c

					s = 0;
					t = (e >= 0 ? 0 : (-e >= c ? 1 : -e / c));
				}
			}
			else
			{
				if (t < 0)
				{
					//region 5
					// F(s) = Q(s, 0) = as^2 + 2ds + f
					// F'(s)/2 = as + d
					// F'(s) = 0 when t = -d/a

					s = (d >= 0 ? 0 : (-d >= a ? 1 : -d / a));
					t = 0;
				}
				else
				{
					//region 0
					Real invDet = 1 / det;
					s *= invDet;
					t *= invDet;
				}
			}
		}
		else
		{
			if (s < 0)
			{
				//region 2
				s = 0;
				t = 1;
			}
			else if (t < 0)
			{
				//region 6
				s = 1;
				t = 0;
			}
			else
			{
				//region 1
				// F(s) = Q(s, 1 - s) = (a - 2b + c)s^2 + 2(b - c + d - e)s + (c + 2e + f)
				// F'(s)/2 = (a - 2b + c)s + (b - c + d - e)
				// F'(s} = 0 when s = (c + e - b - d)/(a - 2b + c)
				// a - 2b + c = |e0 - e1|^2 > 0,
				// so only the sign of c + e - b - d need be considered

				Real numer = c + e - b - d;
				if (numer <= 0) {
					s = 0;
				}
				else {
					Real denom = a - 2 * b + c; // positive quantity
					s = (numer >= denom ? 1 : numer / denom);
				}
				t = 1 - s;
			}
		}
		return TPoint3D<Real>(triangle.v[0] + s * e0 + t * e1);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TRectangle3D<Real>& rectangle) const
	{
		Coord3D diff = origin - rectangle.center;
		Real b0 = diff.dot(rectangle.axis[0]);
		Real b1 = diff.dot(rectangle.axis[1]);
		Coord2D uv(b0, b1);

		uv = clamp(uv, -rectangle.extent, rectangle.extent);

		return TPoint3D<Real>(rectangle.center + uv[0] * rectangle.axis[0] + uv[1] * rectangle.axis[1]);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TDisk3D<Real>& disk) const
	{
		Coord3D cp = origin - disk.center;
		Coord3D cq = cp - cp.dot(disk.normal) * disk.normal;

		Coord3D q;
		q = disk.center + cq;
		if (cq.normSquared() > disk.radius * disk.radius)
		{
			q = disk.center + disk.radius * cq.normalize();
		}
		return TPoint3D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TSphere3D<Real>& sphere) const
	{
		Coord3D cp = origin - sphere.center;
		Coord3D q = sphere.center + sphere.radius * cp.normalize();

		return TPoint3D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TCapsule3D<Real>& capsule) const
	{
		Coord3D coordQ = project(capsule.segment).origin;
		Coord3D dir = origin - coordQ;

		return TPoint3D<Real>(coordQ + capsule.radius * dir.normalize());
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TTet3D<Real>& tet) const
	{
		TPoint3D<Real> closestPt;
		Real minDist = REAL_MAX;
		for (int i = 0; i < 4; i++)
		{
			TTriangle3D<Real> face = tet.face(i);
			TPoint3D<Real> q = project(tet.face(i));
			Real d = (origin - q.origin).normSquared();
			if (d < minDist)
			{
				minDist = d;
				closestPt = q;
			}
		}

		return closestPt;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TAlignedBox3D<Real>& abox) const
	{
		bool bInside = inside(abox);

		if (!bInside)
		{
			return TPoint3D<Real>(clamp(origin, abox.v0, abox.v1));
		}

		//compute the distance to six faces
		Coord3D q;
		Real minDist = REAL_MAX;
		Coord3D offset0 = abs(origin - abox.v0);
		if (offset0[0] < minDist)
		{
			q = Coord3D(abox.v0[0], origin[1], origin[2]);
			minDist = offset0[0];
		}

		if (offset0[1] < minDist)
		{
			q = Coord3D(origin[0], abox.v0[1], origin[2]);
			minDist = offset0[1];
		}

		if (offset0[2] < minDist)
		{
			q = Coord3D(origin[0], origin[1], abox.v0[2]);
			minDist = offset0[2];
		}


		Coord3D offset1 = abs(origin - abox.v1);
		if (offset1[0] < minDist)
		{
			q = Coord3D(abox.v1[0], origin[1], origin[2]);
			minDist = offset1[0];
		}

		if (offset1[1] < minDist)
		{
			q = Coord3D(origin[0], abox.v1[1], origin[2]);
			minDist = offset1[1];
		}

		if (offset1[2] < minDist)
		{
			q = Coord3D(origin[0], origin[1], abox.v1[2]);
			minDist = offset1[2];
		}

		return TPoint3D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TOrientedBox3D<Real>& obb) const
	{
		Coord3D offset = origin - obb.center;
		Coord3D pPrime(offset.dot(obb.u), offset.dot(obb.v), offset.dot(obb.w));

		Coord3D qPrime = TPoint3D<Real>(pPrime).project(TAlignedBox3D<Real>(-obb.extent, obb.extent)).origin;

		Coord3D q = obb.center + qPrime[0] * obb.u + qPrime[1] * obb.v + qPrime[2] * obb.w;

		return TPoint3D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TSphere3D<Real>& sphere, Bool& bInside) const
	{
		Coord3D cp = origin - sphere.center;
		Coord3D q = sphere.center + sphere.radius * cp.normalize();

		bInside = cp.normSquared() >= sphere.radius * sphere.radius ? false : true;

		return TPoint3D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TCapsule3D<Real>& capsule, Bool& bInside /*= Bool(false)*/) const
	{
		Coord3D coordQ = project(capsule.segment).origin;
		Coord3D dir = origin - coordQ;

		bInside = dir.normSquared() < capsule.radius * capsule.radius ? true : false;
		return TPoint3D<Real>(coordQ + capsule.radius * dir.normalize());
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TTet3D<Real>& tet, Bool& bInside) const
	{
		bInside = inside(tet);

		TPoint3D<Real> closestPt;
		Real minDist = REAL_MAX;
		for (int i = 0; i < 4; i++)
		{
			TPoint3D<Real> q = project(tet.face(i));
			Real d = (origin - q.origin).normSquared();
			if (d < minDist)
			{
				minDist = d;
				closestPt = q;
			}
		}

		return closestPt;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TTet3D<Real>& tet, Bool& bInside, int* idx) const
	{
		bInside = true;

		int idx2;
		TPoint3D<Real> closestPt;
		Real minDist = REAL_MAX;
		for (int i = 0; i < 4; i++)
		{
			TTriangle3D<Real> face = tet.face(i);
			TPoint3D<Real> q = project(tet.face(i));
			Real d = (origin - q.origin).normSquared();
			if (d < minDist)
			{
				minDist = d;
				closestPt = q;
				if (i == 0 || i == 3)
					idx2 = 3 - i;
				else
					idx2 = i;
			}
			bInside &= (origin - face.v[0]).dot(face.normal()) < Real(0);
		}

		if (idx != NULL)
		{
			*idx = idx2;
		}

		return closestPt;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TAlignedBox3D<Real>& abox, Bool& bInside) const
	{
		bInside = inside(abox);

		if (!bInside)
		{
			return TPoint3D<Real>(clamp(origin, abox.v0, abox.v1));
		}

		//compute the distance to six faces
		Coord3D q;
		Real minDist = REAL_MAX;
		Coord3D offset0 = abs(origin - abox.v0);
		if (offset0[0] < minDist)
		{
			q = Coord3D(abox.v0[0], origin[1], origin[2]);
			minDist = offset0[0];
		}

		if (offset0[1] < minDist)
		{
			q = Coord3D(origin[0], abox.v0[1], origin[2]);
			minDist = offset0[1];
		}

		if (offset0[2] < minDist)
		{
			q = Coord3D(origin[0], origin[1], abox.v0[2]);
			minDist = offset0[2];
		}


		Coord3D offset1 = abs(origin - abox.v1);
		if (offset1[0] < minDist)
		{
			q = Coord3D(abox.v1[0], origin[1], origin[2]);
			minDist = offset1[0];
		}

		if (offset1[1] < minDist)
		{
			q = Coord3D(origin[0], abox.v1[1], origin[2]);
			minDist = offset1[1];
		}

		if (offset1[2] < minDist)
		{
			q = Coord3D(origin[0], origin[1], abox.v1[2]);
			minDist = offset1[2];
		}

		return TPoint3D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPoint3D<Real>::project(const TOrientedBox3D<Real>& obb, Bool& bInside) const
	{
		Coord3D offset = origin - obb.center;
		Coord3D pPrime(offset.dot(obb.u), offset.dot(obb.v), offset.dot(obb.w));

		Coord3D qPrime = TPoint3D<Real>(pPrime).project(TAlignedBox3D<Real>(-obb.extent, obb.extent), bInside).origin;

		Coord3D q = obb.center + qPrime[0] * obb.u + qPrime[1] * obb.v + qPrime[2] * obb.w;

		return TPoint3D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TPoint3D<Real>& pt) const
	{
		return (origin - pt.origin).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TLine3D<Real>& line) const
	{
		return (origin - project(line).origin).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TRay3D<Real>& ray) const
	{
		return (origin - project(ray).origin).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TSegment3D<Real>& segment) const
	{
		return (origin - project(segment).origin).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TPlane3D<Real>& plane) const
	{
		Coord3D q = project(plane).origin;
		Real sign = (origin - q).dot(plane.normal) < Real(0) ? Real(-1) : Real(1);

		return sign * (origin - q).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TTriangle3D<Real>& triangle) const
	{
		Coord3D q = project(triangle).origin;
		Real sign;
		Real sign0 = (origin - q).dot(triangle.normal());
		if (abs(sign0) < Real(REAL_EPSILON) && (origin - q).norm() > Real(REAL_EPSILON))
			sign = Real(1);
		else
			sign = sign0 < Real(0) ? Real(-1) : Real(1);
		//Real sign = (origin - q).dot(triangle.normal()) < Real(0) ? Real(-1) : Real(1);

		return sign * (origin - q).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TRectangle3D<Real>& rectangle) const
	{
		Coord3D q = project(rectangle).origin;
		Real sign = (origin - q).dot(rectangle.normal()) < Real(0) ? Real(-1) : Real(1);

		return (origin - q).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TDisk3D<Real>& disk) const
	{
		Coord3D q = project(disk).origin;
		Real sign = (origin - q).dot(disk.normal) < Real(0) ? Real(-1) : Real(1);

		return (origin - q).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TSphere3D<Real>& sphere) const
	{
		return (origin - sphere.center).norm() - sphere.radius;
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TTet3D<Real>& tet) const
	{
		Bool bInside;
		Real d = (origin - project(tet, bInside).origin).norm();
		return bInside == true ? -d : d;
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TAlignedBox3D<Real>& abox) const
	{
		Bool bInside;
		Real d = (origin - project(abox, bInside).origin).norm();
		return bInside == true ? -d : d;
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TOrientedBox3D<Real>& obb) const
	{
		Bool bInside;
		Real d = (origin - project(obb, bInside).origin).norm();
		return bInside == true ? -d : d;
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distance(const TCapsule3D<Real>& capsule) const
	{
		Bool bInside;
		Real d = (origin - project(capsule, bInside).origin).norm();
		return bInside == true ? -d : d;
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TPoint3D& pt) const
	{
		return (origin - pt.origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TLine3D<Real>& line) const
	{
		return (origin - project(line).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TRay3D<Real>& ray) const
	{
		return (origin - project(ray).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TSegment3D<Real>& segment) const
	{
		return (origin - project(segment).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TPlane3D<Real>& plane) const
	{
		return (origin - project(plane).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TTriangle3D<Real>& triangle) const
	{
		return (origin - project(triangle).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TRectangle3D<Real>& rectangle) const
	{
		return (origin - project(rectangle).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TDisk3D<Real>& disk) const
	{
		return (origin - project(disk).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TSphere3D<Real>& sphere) const
	{
		return (origin - project(sphere).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TAlignedBox3D<Real>& abox) const
	{
		return (origin - project(abox).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TOrientedBox3D<Real>& obb) const
	{
		return (origin - project(obb).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TTet3D<Real>& tet) const
	{
		return (origin - project(tet).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint3D<Real>::distanceSquared(const TCapsule3D<Real>& capsule) const
	{
		return (origin - project(capsule).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TLine3D<Real>& line) const
	{
		if (!line.isValid())
		{
			return false;
		}

		return (origin - line.origin).cross(line.direction).normSquared() < REAL_EPSILON_SQUARED;
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TRay3D<Real>& ray) const
	{
		if (!inside(TLine3D<Real>(ray.origin, ray.direction)))
		{
			return false;
		}

		Coord3D offset = origin - ray.origin;
		Real t = offset.dot(ray.direction);

		return t > Real(0);
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TSegment3D<Real>& segment) const
	{
		Coord3D dir = segment.direction();
		if (!inside(TLine3D<Real>(segment.startPoint(), dir)))
		{
			return false;
		}

		Coord3D offset = origin - segment.startPoint();
		Real t = offset.dot(dir) / dir.normSquared();

		return t > Real(0) && t < Real(1);
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TPlane3D<Real>& plane) const
	{
		if (!plane.isValid())
		{
			return false;
		}

		return abs((origin - plane.origin).dot(plane.normal)) < REAL_EPSILON;
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TTriangle3D<Real>& triangle) const
	{
		TPlane3D<Real> plane(triangle.v[0], triangle.normal());
		if (!inside(plane))
		{
			return false;
		}

		typename TTriangle3D<Real>::Param tParam;
		bool bValid = triangle.computeBarycentrics(origin, tParam);
		if (bValid)
		{
			return tParam.u > Real(0) && tParam.u < Real(1) && tParam.v > Real(0) && tParam.v < Real(1) && tParam.w > Real(0) && tParam.w < Real(1);
		}
		else
		{
			return false;
		}
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TRectangle3D<Real>& rectangle) const
	{
		TPlane3D<Real> plane(rectangle.center, rectangle.normal());
		if (!inside(plane))
		{
			return false;
		}

		typename TRectangle3D<Real>::Param recParam;
		bool bValid = rectangle.computeParams(origin, recParam);
		if (bValid)
		{
			return recParam.u < rectangle.extent[0] && recParam.u > -rectangle.extent[0] && recParam.v < rectangle.extent[1] && recParam.v > -rectangle.extent[1];
		}
		else
		{
			return false;
		}
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TDisk3D<Real>& disk) const
	{
		TPlane3D<Real> plane(disk.center, disk.normal);
		if (!inside(plane))
		{
			return false;
		}

		return (origin - disk.center).normSquared() < disk.radius * disk.radius;
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TSphere3D<Real>& sphere) const
	{
		return (origin - sphere.center).normSquared() < sphere.radius * sphere.radius;
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TCapsule3D<Real>& capsule) const
	{
		return distanceSquared(capsule.segment) < capsule.radius * capsule.radius;
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TTet3D<Real>& tet) const
	{
		bool bInside = true;

		TTriangle3D<Real> face;
		Coord3D normal;
		face = tet.face(0);
		normal = face.normal();
		bInside &= (origin - face.v[0]).dot(normal) * (tet.v[0] - face.v[0]).dot(normal) > 0;

		face = tet.face(1);
		normal = face.normal();
		bInside &= (origin - face.v[0]).dot(normal) * (tet.v[1] - face.v[0]).dot(normal) > 0;

		face = tet.face(2);
		normal = face.normal();
		bInside &= (origin - face.v[0]).dot(normal) * (tet.v[2] - face.v[0]).dot(normal) > 0;

		face = tet.face(3);
		normal = face.normal();
		bInside &= (origin - face.v[0]).dot(normal) * (tet.v[3] - face.v[0]).dot(normal) > 0;

		return bInside;
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TOrientedBox3D<Real>& obb) const
	{
		Coord3D offset = origin - obb.center;
		Coord3D pPrime(offset.dot(obb.u), offset.dot(obb.v), offset.dot(obb.w));

		bool bInside = true;
		bInside &= pPrime[0] < obb.extent[0] && pPrime[0] >  -obb.extent[0];
		bInside &= pPrime[1] < obb.extent[1] && pPrime[1] >  -obb.extent[1];
		bInside &= pPrime[2] < obb.extent[2] && pPrime[2] >  -obb.extent[2];
		return bInside;
	}

	template<typename Real>
	DYN_FUNC bool TPoint3D<Real>::inside(const TAlignedBox3D<Real>& box) const
	{
		Coord3D offset = origin - box.v0;
		Coord3D extent = box.v1 - box.v0;

		bool bInside = true;
		bInside &= offset[0] < extent[0] && offset[0] >  Real(0);
		bInside &= offset[1] < extent[1] && offset[1] >  Real(0);
		bInside &= offset[2] < extent[2] && offset[2] >  Real(0);

		return bInside;
	}

	template<typename Real>
	DYN_FUNC const TSegment3D<Real> TPoint3D<Real>::operator-(const TPoint3D<Real>& pt) const
	{
		return TSegment3D<Real>(pt.origin, origin);
	}

	template<typename Real>
	DYN_FUNC TLine3D<Real>::TLine3D()
	{
		origin = Coord3D(0);
		direction = Coord3D(1, 0, 0);
	}

	template<typename Real>
	DYN_FUNC TLine3D<Real>::TLine3D(const Coord3D& pos, const Coord3D& dir)
	{
		origin = pos;
		direction = dir;
	}

	template<typename Real>
	DYN_FUNC TLine3D<Real>::TLine3D(const TLine3D<Real>& line)
	{
		origin = line.origin;
		direction = line.direction;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TLine3D<Real>::proximity(const TLine3D<Real>& line) const
	{
		Coord3D u = origin - line.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(line.direction);
		Real c = line.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(line.direction);
		Real f = u.normSquared();
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			TPoint3D<Real> p = TPoint3D<Real>(line.origin);
			return c < REAL_EPSILON ? p - p.project(*this) : TSegment3D<Real>(origin, line.origin + e / c * line.direction);
		}
		else
		{
			Real invDet = 1 / det;
			Real s = (b * e - c * d) * invDet;
			Real t = (a * e - b * d) * invDet;
			return TSegment3D<Real>(origin + s * direction, line.origin + t * line.direction);
		}
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TLine3D<Real>::proximity(const TRay3D<Real>& ray) const
	{
		Coord3D u = origin - ray.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(ray.direction);
		Real c = ray.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(ray.direction);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			TPoint3D<Real> p0(origin);
			TPoint3D<Real> p1(ray.origin);

			return a < REAL_EPSILON ? p0.project(*this) - p0 : p1 - p1.project(*this);
		}

		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		if (tNum < 0) {
			tNum = 0;
			sNum = -d;
			sDenom = a;
		}
		// Parameters of nearest points on restricted domain
		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return TSegment3D<Real>(origin + (s * direction), ray.origin + (t * ray.direction));
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TLine3D<Real>::proximity(const TSegment3D<Real>& segment) const
	{
		Coord3D u = origin - segment.startPoint();
		Coord3D dir1 = segment.endPoint() - segment.startPoint();
		Real a = direction.normSquared();
		Real b = direction.dot(dir1);
		Real c = dir1.dot(dir1);
		Real d = u.dot(direction);
		Real e = u.dot(dir1);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			TPoint3D<Real> p0(origin);
			TPoint3D<Real> p1(segment.startPoint());

			return a < REAL_EPSILON ? p0.project(*this) - p0 : p1 - p1.project(*this);
		}

		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		// Check t
		if (tNum < 0) {
			tNum = 0;
			sNum = -d;
			sDenom = a;
		}
		else if (tNum > tDenom) {
			tNum = tDenom;
			sNum = -d + b;
			sDenom = a;
		}
		// Parameters of nearest points on restricted domain
		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return TSegment3D<Real>(origin + (s * direction), segment.startPoint() + (t * dir1));
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TLine3D<Real>::proximity(const TTriangle3D<Real>& triangle) const
	{
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Coord3D normal = e0.cross(e1);
		Real NdD = normal.dot(direction);
		if (abs(NdD) > REAL_EPSILON)
		{
			// The line and triangle are not parallel, so the line
			// intersects/ the plane of the triangle.
			Coord3D diff = origin - triangle.v[0];

			//compute a right - handed orthonormal basis
			Coord3D W = direction;
			W.normalize();
			Coord3D U = W[0] > W[1] ? Coord3D(-W[2], (Real)0, W[0]) : Coord3D((Real)0, W[2], -W[1]);
			Coord3D V = W.cross(U);

			Real UdE0 = U.dot(e0);
			Real UdE1 = U.dot(e1);
			Real UdDiff = U.dot(diff);
			Real VdE0 = V.dot(e0);
			Real VdE1 = V.dot(e1);
			Real VdDiff = V.dot(diff);
			Real invDet = ((Real)1) / (UdE0 * VdE1 - UdE1 * VdE0);

			// Barycentric coordinates for the point of intersection.
			Real u = (VdE1 * UdDiff - UdE1 * VdDiff) * invDet;
			Real v = (UdE0 * VdDiff - VdE0 * UdDiff) * invDet;
			Real b0 = (Real)1 - u - v;

			if (b0 >= (Real)0 && u >= (Real)0 && v >= (Real)0)
			{
				// Line parameter for the point of intersection.
				Real DdE0 = W.dot(e0);
				Real DdE1 = W.dot(e1);
				Real DdDiff = W.dot(diff);
				Real t = u * DdE0 + v * DdE1 - DdDiff;

				return TSegment3D<Real>(origin + t * W, triangle.v[0] + u * e0 + v * e1);
			}
		}

		if (direction.normSquared() < EPSILON)
		{
			TPoint3D<Real> p(origin);
			return p.project(triangle) - p;
		}
		else
		{
			// Treat line and triangle as parallel. Compute
			// closest points by computing distance from
			// line to each triangle edge and taking minimum.
			TSegment3D<Real> e0(triangle.v[0], triangle.v[1]);
			TSegment3D<Real> minPQ = proximity(e0);
			Real minDS = minPQ.lengthSquared();

			TSegment3D<Real> e1(triangle.v[0], triangle.v[2]);
			TSegment3D<Real> pq1 = proximity(e1);
			if (pq1.lengthSquared() < minDS)
			{
				minPQ = pq1;
				minDS = pq1.lengthSquared();
			}

			TSegment3D<Real> e2(triangle.v[1], triangle.v[2]);
			TSegment3D<Real> pq2 = proximity(e2);
			if (pq2.lengthSquared() < minDS)
			{
				minPQ = pq2;
			}

			return minPQ;
		}
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TLine3D<Real>::proximity(const TRectangle3D<Real>& rectangle) const
	{
		Coord3D N = rectangle.axis[0].cross(rectangle.axis[1]);
		Real NdD = N.dot(direction);
		if (abs(NdD) > REAL_EPSILON)
		{
			// The line and rectangle are not parallel, so the line
			// intersects the plane of the rectangle.
			Coord3D diff = origin - rectangle.center;
			Coord3D W = direction;
			W.normalize();
			Coord3D U = W[0] > W[1] ? Coord3D(-W[2], (Real)0, W[0]) : Coord3D((Real)0, W[2], -W[1]);
			Coord3D V = W.cross(U);
			Real UdD0 = U.dot(rectangle.axis[0]);
			Real UdD1 = U.dot(rectangle.axis[1]);
			Real UdPmC = U.dot(diff);
			Real VdD0 = V.dot(rectangle.axis[0]);
			Real VdD1 = V.dot(rectangle.axis[1]);
			Real VdPmC = V.dot(diff);
			Real invDet = ((Real)1) / (UdD0 * VdD1 - UdD1 * VdD0);

			// Rectangle coordinates for the point of intersection.
			Real s0 = (VdD1 * UdPmC - UdD1 * VdPmC) * invDet;
			Real s1 = (UdD0 * VdPmC - VdD0 * UdPmC) * invDet;

			if (abs(s0) <= rectangle.extent[0] && abs(s1) <= rectangle.extent[1])
			{
				// Line parameter for the point of intersection.
				Real DdD0 = direction.dot(rectangle.axis[0]);
				Real DdD1 = direction.dot(rectangle.axis[1]);
				Real DdDiff = direction.dot(diff);
				Real t = s0 * DdD0 + s1 * DdD1 - DdDiff;

				return TSegment3D<Real>(origin + t * direction, rectangle.center + s0 * rectangle.axis[0] + s1 * rectangle.axis[1]);
			}
		}

		Real minDS = REAL_MAX;
		TSegment3D<Real> minPQ;

		for (int i = 0; i < 4; i++)
		{
			TSegment3D<Real> pq = proximity(rectangle.edge(i));
			if (pq.lengthSquared() < minDS)
			{
				minDS = pq.lengthSquared();
				minPQ = pq;
			}
		}

		return minPQ;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TLine3D<Real>::proximity(const TSphere3D<Real>& sphere) const
	{
		Coord3D offset = sphere.center - origin;
		Real d2 = direction.normSquared();
		if (d2 < REAL_EPSILON)
		{
			return TPoint3D<Real>(origin).project(sphere) - TPoint3D<Real>(origin);
		}

		Coord3D p = origin + offset.dot(direction) / d2 * direction;

		return TPoint3D<Real>(p).project(sphere) - TPoint3D<Real>(p);
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TLine3D<Real>::proximity(const TAlignedBox3D<Real>& box) const
	{
		Coord3D boxCenter, boxExtent;
		boxCenter = 0.5 * (box.v0 + box.v1);
		boxExtent = 0.5 * (box.v1 - box.v0);
		Coord3D point = origin - boxCenter;
		Coord3D lineDir = direction;

		Real t = Real(0);

		auto case00 = [](Coord3D& pt, Real& t, const Coord3D dir, const Coord3D extent, const int axis) {
			t = (extent[axis] - pt[axis]) / dir[axis];
			pt[axis] = extent[axis];

			pt = clamp(pt, -extent, extent);
		};

		auto case0 = [](Coord3D& pt, Real& t, const Coord3D dir, const Coord3D extent, const int i0, const int i1, const int i2) {
			Real PmE0 = pt[i0] - extent[i0];
			Real PmE1 = pt[i1] - extent[i1];
			Real prod0 = dir[i1] * PmE0;
			Real prod1 = dir[i0] * PmE1;

			if (prod0 >= prod1)
			{
				// line intersects P[i0] = e[i0]
				pt[i0] = extent[i0];

				Real PpE1 = pt[i1] + extent[i1];
				Real delta = prod0 - dir[i0] * PpE1;
				if (delta >= (Real)0)
				{
					Real invLSqr = ((Real)1) / (dir[i0] * dir[i0] + dir[i1] * dir[i1]);
					pt[i1] = -extent[i1];
					t = -(dir[i0] * PmE0 + dir[i1] * PpE1) * invLSqr;
				}
				else
				{
					Real inv = ((Real)1) / dir[i0];
					pt[i1] -= prod0 * inv;
					t = -PmE0 * inv;
				}
			}
			else
			{
				// line intersects P[i1] = e[i1]
				pt[i1] = extent[i1];

				Real PpE0 = pt[i0] + extent[i0];
				Real delta = prod1 - dir[i1] * PpE0;
				if (delta >= (Real)0)
				{
					Real invLSqr = ((Real)1) / (dir[i0] * dir[i0] + dir[i1] * dir[i1]);
					pt[i0] = -extent[i0];
					t = -(dir[i0] * PpE0 + dir[i1] * PmE1) * invLSqr;
				}
				else
				{
					Real inv = ((Real)1) / dir[i1];
					pt[i0] -= prod1 * inv;
					t = -PmE1 * inv;
				}
			}

			if (pt[i2] < -extent[i2])
			{
				pt[i2] = -extent[i2];
			}
			else if (pt[i2] > extent[i2])
			{
				pt[i2] = extent[i2];
			}
		};

		auto face = [](Coord3D& pnt, Real& final_t,
			const Coord3D& dir, const Coord3D& PmE, const Coord3D& boxExtent,
			int i0, int i1, int i2)
		{
			Coord3D PpE;
			Real lenSqr, inv, tmp, param, t, delta;

			PpE[i1] = pnt[i1] + boxExtent[i1];
			PpE[i2] = pnt[i2] + boxExtent[i2];
			if (dir[i0] * PpE[i1] >= dir[i1] * PmE[i0])
			{
				if (dir[i0] * PpE[i2] >= dir[i2] * PmE[i0])
				{
					// v[i1] >= -e[i1], v[i2] >= -e[i2] (distance = 0)
					pnt[i0] = boxExtent[i0];
					inv = ((Real)1) / dir[i0];
					pnt[i1] -= dir[i1] * PmE[i0] * inv;
					pnt[i2] -= dir[i2] * PmE[i0] * inv;
					final_t = -PmE[i0] * inv;
				}
				else
				{
					// v[i1] >= -e[i1], v[i2] < -e[i2]
					lenSqr = dir[i0] * dir[i0] + dir[i2] * dir[i2];
					tmp = lenSqr * PpE[i1] - dir[i1] * (dir[i0] * PmE[i0] +
						dir[i2] * PpE[i2]);
					if (tmp <= ((Real)2) * lenSqr * boxExtent[i1])
					{
						t = tmp / lenSqr;
						lenSqr += dir[i1] * dir[i1];
						tmp = PpE[i1] - t;
						delta = dir[i0] * PmE[i0] + dir[i1] * tmp + dir[i2] * PpE[i2];
						param = -delta / lenSqr;

						final_t = param;
						pnt[i0] = boxExtent[i0];
						pnt[i1] = t - boxExtent[i1];
						pnt[i2] = -boxExtent[i2];
					}
					else
					{
						lenSqr += dir[i1] * dir[i1];
						delta = dir[i0] * PmE[i0] + dir[i1] * PmE[i1] + dir[i2] * PpE[i2];
						param = -delta / lenSqr;

						final_t = param;
						pnt[i0] = boxExtent[i0];
						pnt[i1] = boxExtent[i1];
						pnt[i2] = -boxExtent[i2];
					}
				}
			}
			else
			{
				if (dir[i0] * PpE[i2] >= dir[i2] * PmE[i0])
				{
					// v[i1] < -e[i1], v[i2] >= -e[i2]
					lenSqr = dir[i0] * dir[i0] + dir[i1] * dir[i1];
					tmp = lenSqr * PpE[i2] - dir[i2] * (dir[i0] * PmE[i0] +
						dir[i1] * PpE[i1]);
					if (tmp <= ((Real)2) * lenSqr * boxExtent[i2])
					{
						t = tmp / lenSqr;
						lenSqr += dir[i2] * dir[i2];
						tmp = PpE[i2] - t;
						delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * tmp;
						param = -delta / lenSqr;

						final_t = param;
						pnt[i0] = boxExtent[i0];
						pnt[i1] = -boxExtent[i1];
						pnt[i2] = t - boxExtent[i2];
					}
					else
					{
						lenSqr += dir[i2] * dir[i2];
						delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * PmE[i2];
						param = -delta / lenSqr;

						final_t = param;
						pnt[i0] = boxExtent[i0];
						pnt[i1] = -boxExtent[i1];
						pnt[i2] = boxExtent[i2];
					}
				}
				else
				{
					// v[i1] < -e[i1], v[i2] < -e[i2]
					lenSqr = dir[i0] * dir[i0] + dir[i2] * dir[i2];
					tmp = lenSqr * PpE[i1] - dir[i1] * (dir[i0] * PmE[i0] +
						dir[i2] * PpE[i2]);
					if (tmp >= (Real)0)
					{
						// v[i1]-edge is closest
						if (tmp <= ((Real)2) * lenSqr * boxExtent[i1])
						{
							t = tmp / lenSqr;
							lenSqr += dir[i1] * dir[i1];
							tmp = PpE[i1] - t;
							delta = dir[i0] * PmE[i0] + dir[i1] * tmp + dir[i2] * PpE[i2];
							param = -delta / lenSqr;

							final_t = param;
							pnt[i0] = boxExtent[i0];
							pnt[i1] = t - boxExtent[i1];
							pnt[i2] = -boxExtent[i2];
						}
						else
						{
							lenSqr += dir[i1] * dir[i1];
							delta = dir[i0] * PmE[i0] + dir[i1] * PmE[i1]
								+ dir[i2] * PpE[i2];
							param = -delta / lenSqr;

							final_t = param;
							pnt[i0] = boxExtent[i0];
							pnt[i1] = boxExtent[i1];
							pnt[i2] = -boxExtent[i2];
						}
						return;
					}

					lenSqr = dir[i0] * dir[i0] + dir[i1] * dir[i1];
					tmp = lenSqr * PpE[i2] - dir[i2] * (dir[i0] * PmE[i0] +
						dir[i1] * PpE[i1]);
					if (tmp >= (Real)0)
					{
						// v[i2]-edge is closest
						if (tmp <= ((Real)2) * lenSqr * boxExtent[i2])
						{
							t = tmp / lenSqr;
							lenSqr += dir[i2] * dir[i2];
							tmp = PpE[i2] - t;
							delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * tmp;
							param = -delta / lenSqr;

							final_t = param;
							pnt[i0] = boxExtent[i0];
							pnt[i1] = -boxExtent[i1];
							pnt[i2] = t - boxExtent[i2];
						}
						else
						{
							lenSqr += dir[i2] * dir[i2];
							delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1]
								+ dir[i2] * PmE[i2];
							param = -delta / lenSqr;

							final_t = param;
							pnt[i0] = boxExtent[i0];
							pnt[i1] = -boxExtent[i1];
							pnt[i2] = boxExtent[i2];
						}
						return;
					}

					// (v[i1],v[i2])-corner is closest
					lenSqr += dir[i2] * dir[i2];
					delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * PpE[i2];
					param = -delta / lenSqr;

					final_t = param;
					pnt[i0] = boxExtent[i0];
					pnt[i1] = -boxExtent[i1];
					pnt[i2] = -boxExtent[i2];
				}
			}
		};


		// Apply reflections so that direction vector has nonnegative
		// components.
		bool reflect[3];
		for (int i = 0; i < 3; ++i)
		{
			if (lineDir[i] < (Real)0)
			{
				point[i] = -point[i];
				lineDir[i] = -lineDir[i];
				reflect[i] = true;
			}
			else
			{
				reflect[i] = false;
			}
		}

		if (lineDir[0] > REAL_EPSILON)
		{
			if (lineDir[1] > REAL_EPSILON)
			{
				if (lineDir[2] > REAL_EPSILON)  // (+,+,+)
				{
					//CaseNoZeros(point, dir, boxExtent, t);
					Coord3D PmE = point - boxExtent;
					Real prodDxPy = lineDir[0] * PmE[1];
					Real prodDyPx = lineDir[1] * PmE[0];
					Real prodDzPx, prodDxPz, prodDzPy, prodDyPz;

					if (prodDyPx >= prodDxPy)
					{
						prodDzPx = lineDir[2] * PmE[0];
						prodDxPz = lineDir[0] * PmE[2];
						if (prodDzPx >= prodDxPz)
						{
							// line intersects x = e0
							face(point, t, lineDir, PmE, boxExtent, 0, 1, 2);
						}
						else
						{
							// line intersects z = e2
							face(point, t, lineDir, PmE, boxExtent, 2, 0, 1);
						}
					}
					else
					{
						prodDzPy = lineDir[2] * PmE[1];
						prodDyPz = lineDir[1] * PmE[2];
						if (prodDzPy >= prodDyPz)
						{
							// line intersects y = e1
							face(point, t, lineDir, PmE, boxExtent, 1, 2, 0);
						}
						else
						{
							// line intersects z = e2
							face(point, t, lineDir, PmE, boxExtent, 2, 0, 1);
						}
					}
				}
				else  // (+,+,0)
				{
					//Case0(0, 1, 2, point, dir, boxExtent, t);
					case0(point, t, lineDir, boxExtent, 0, 1, 2);
				}
			}
			else
			{
				if (lineDir[2] > REAL_EPSILON)  // (+,0,+)
				{
					//Case0(0, 2, 1, point, dir, boxExtent, t);
					case0(point, t, lineDir, boxExtent, 0, 2, 1);
				}
				else  // (+,0,0)
				{
					//Case00(0, 1, 2, point, dir, boxExtent, t);
					case00(point, t, lineDir, boxExtent, 0);
				}
			}
		}
		else
		{
			if (lineDir[1] > REAL_EPSILON)
			{
				if (lineDir[2] > REAL_EPSILON)  // (0,+,+)
				{
					//Case0(1, 2, 0, point, dir, boxExtent, t);
					case0(point, t, lineDir, boxExtent, 1, 2, 0);
				}
				else  // (0,+,0)
				{
					//Case00(1, 0, 2, point, dir, boxExtent, t);
					case00(point, t, lineDir, boxExtent, 1);
				}
			}
			else
			{
				if (lineDir[2] > REAL_EPSILON)  // (0,0,+)
				{
					//Case00(2, 0, 1, point, dir, boxExtent, t);
					case00(point, t, lineDir, boxExtent, 2);
				}
				else  // (0,0,0)
				{
					point = clamp(point, -boxExtent, boxExtent);
					//Case000(point, boxExtent);
				}
			}
		}

		// Undo the reflections applied previously.
		for (int i = 0; i < 3; ++i)
		{
			if (reflect[i])
			{
				point[i] = -point[i];
			}
		}

		return TSegment3D<Real>(origin + t * direction, boxCenter + point);
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TLine3D<Real>::proximity(const TOrientedBox3D<Real>& obb) const
	{
		//transform to the local coordinate system of obb
		Coord3D diff = origin - obb.center;
		Coord3D originPrime = Coord3D(diff.dot(obb.u), diff.dot(obb.v), diff.dot(obb.w));
		Coord3D dirPrime = Coord3D(direction.dot(obb.u), direction.dot(obb.v), direction.dot(obb.w));

		TSegment3D<Real> pqPrime = TLine3D<Real>(originPrime, dirPrime).proximity(TAlignedBox3D<Real>(obb.center - obb.extent, obb.center + obb.extent));

		Coord3D pPrime = pqPrime.startPoint();
		Coord3D qPrime = pqPrime.endPoint();

		//transform back to the global coordinate system
		Coord3D p = pPrime[0] * obb.u + pPrime[1] * obb.v + pPrime[2] * obb.w;
		Coord3D q = qPrime[0] * obb.u + qPrime[1] * obb.v + qPrime[2] * obb.w;

		return TSegment3D<Real>(p, q);
	}

	// 	DYN_FUNC Segment3D Line3D::proximity(const Segment3D& segment) const
// 	{
// 
// 	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distance(const TPoint3D<Real>& pt) const
	{
		return pt.distance(*this);
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distance(const TLine3D<Real>& line) const
	{
		return proximity(line).length();
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distance(const TRay3D<Real>& ray) const
	{
		return proximity(ray).length();
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distance(const TSegment3D<Real>& segment) const
	{
		return proximity(segment).length();
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distance(const TAlignedBox3D<Real>& box) const
	{
		return proximity(box).length();
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distance(const TOrientedBox3D<Real>& obb) const
	{
		return proximity(obb).length();
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distanceSquared(const TPoint3D<Real>& pt) const
	{
		return pt.distanceSquared(*this);
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distanceSquared(const TLine3D<Real>& line) const
	{
		return proximity(line).lengthSquared();
		// 		Coord3D u = origin - line.origin;
		// 		Real a = direction.normSquared();
		// 		Real b = direction.dot(line.direction);
		// 		Real c = line.direction.normSquared();
		// 		Real d = u.dot(direction);
		// 		Real e = u.dot(line.direction);
		// 		Real f = u.normSquared();
		// 		Real det = a * c - b * b;
		// 
		// 		if (det < REAL_EPSILON)
		// 		{
		// 			return f - e * e;
		// 		}
		// 		else
		// 		{
		// 			Real invDet = 1 / det;
		// 			Real s = (b * e - c * d) * invDet;
		// 			Real t = (a * e - b * d) * invDet;
		// 			return s * (a * s - b * t + 2*d) - t * (b * s - c * t + 2*e) + f;
		// 		}
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distanceSquared(const TRay3D<Real>& ray) const
	{
		return proximity(ray).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distanceSquared(const TSegment3D<Real>& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distanceSquared(const TAlignedBox3D<Real>& box) const
	{
		return proximity(box).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::distanceSquared(const TOrientedBox3D<Real>& obb) const
	{
		return proximity(obb).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC int TLine3D<Real>::intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const
	{
		Real DdN = direction.dot(plane.normal);
		if (abs(DdN) < REAL_EPSILON)
		{
			return 0;
		}

		Coord3D offset = origin - plane.origin;
		Real t = -offset.dot(plane.normal) / DdN;

		interPt.origin = origin + t * direction;
		return 1;
	}

	template<typename Real>
	DYN_FUNC int TLine3D<Real>::intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const
	{
		Coord3D diff = origin - triangle.v[0];
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Coord3D normal = e0.cross(e1);

		Real DdN = direction.dot(normal);
		Real sign;
		if (DdN >= REAL_EPSILON)
		{
			sign = Real(1);
		}
		else if (DdN <= -REAL_EPSILON)
		{
			sign = (Real)-1;
			DdN = -DdN;
		}
		else
		{
			return 0;
		}

		Real DdQxE1 = sign * direction.dot(diff.cross(e1));
		if (DdQxE1 >= (Real)0)
		{
			Real DdE0xQ = sign * direction.dot(e0.cross(diff));
			if (DdE0xQ >= (Real)0)
			{
				if (DdQxE1 + DdE0xQ <= DdN)
				{
					// Line intersects triangle.
					Real QdN = -sign * diff.dot(normal);
					Real inv = (Real)1 / DdN;

					Real t = QdN * inv;
					interPt.origin = origin + t * direction;
					return 1;
				}
				// else: b1+b2 > 1, no intersection
			}
			// else: b2 < 0, no intersection
		}
		// else: b1 < 0, no intersection

		return 0;
	}

	template<typename Real>
	DYN_FUNC int TLine3D<Real>::intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const
	{
		Coord3D diff = origin - sphere.center;
		Real a0 = diff.dot(diff) - sphere.radius * sphere.radius;
		Real a1 = direction.dot(diff);

		// Intersection occurs when Q(t) has real roots.
		Real discr = a1 * a1 - a0;
		if (discr > (Real)0)
		{
			Real root = glm::sqrt(discr);
			interSeg.startPoint() = origin + (-a1 - root) * direction;
			interSeg.endPoint() = origin + (-a1 + root) * direction;
			return 2;
		}
		else if (discr < (Real)0)
		{
			return 0;
		}
		else
		{
			interSeg.startPoint() = origin - a1 * direction;
			return 1;
		}
	}

	//TODO:
	template<typename Real>
	DYN_FUNC int TLine3D<Real>::intersect(const TTet3D<Real>& tet, TSegment3D<Real>& interSeg) const
	{
		Point3D p1, p2;

		bool tmp1 = false;
		for (int i = 0; i < 4; i++)
			if (tmp1 == false)
			{
				if (intersect(tet.face(i), p1))
					tmp1 = true;
			}
			else if (intersect(tet.face(i), p2))
			{

				if (p2.distance(p1) > EPSILON)
				{
					interSeg = Segment3D(p1.origin, p2.origin);
					return true;
				}
			}

		return 0;
	}

	template<typename Real>
	DYN_FUNC int TLine3D<Real>::intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const
	{
		if (!isValid())
		{
			return 0;
		}

		Real t0 = -REAL_MAX;
		Real t1 = REAL_MAX;

		auto clip = [](Real denom, Real numer, Real& t0, Real& t1) -> bool
		{
			if (denom > REAL_EPSILON)
			{
				if (numer > denom * t1)
				{
					return false;
				}
				if (numer > denom * t0)
				{
					t0 = numer / denom;
				}
				return true;
			}
			else if (denom < -REAL_EPSILON)
			{
				if (numer > denom * t0)
				{
					return false;
				}
				if (numer > denom * t1)
				{
					t1 = numer / denom;
				}
				return true;
			}
			else
			{
				return numer <= -REAL_EPSILON;
			}
		};

		Coord3D boxCenter = 0.5 * (abox.v0 + abox.v1);
		Coord3D boxExtent = 0.5 * (abox.v1 - abox.v0);

		Coord3D offset = origin - boxCenter;
		Coord3D lineDir = direction;
		lineDir.normalize();

		if (clip(+lineDir[0], -offset[0] - boxExtent[0], t0, t1) &&
			clip(-lineDir[0], +offset[0] - boxExtent[0], t0, t1) &&
			clip(+lineDir[1], -offset[1] - boxExtent[1], t0, t1) &&
			clip(-lineDir[1], +offset[1] - boxExtent[1], t0, t1) &&
			clip(+lineDir[2], -offset[2] - boxExtent[2], t0, t1) &&
			clip(-lineDir[2], +offset[2] - boxExtent[2], t0, t1))
		{
			if (t1 > t0)
			{
				interSeg.v0 = origin + t0 * lineDir;
				interSeg.v1 = origin + t1 * lineDir;
				return 2;
			}
			else
			{
				interSeg.v0 = origin + t0 * lineDir;
				interSeg.v1 = interSeg.v0;
				return 1;
			}
		}

		return 0;
	}

	template<typename Real>
	DYN_FUNC Real TLine3D<Real>::parameter(const Coord3D& pos) const
	{
		Coord3D l = pos - origin;
		Real d2 = direction.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(direction) / d2;
	}

	template<typename Real>
	DYN_FUNC bool TLine3D<Real>::isValid() const
	{
		return direction.normSquared() > REAL_EPSILON_SQUARED;
	}

	template<typename Real>
	DYN_FUNC TRay3D<Real>::TRay3D()
	{
		origin = Coord3D(0);
		direction = Coord3D(1, 0, 0);
	}

	template<typename Real>
	DYN_FUNC TRay3D<Real>::TRay3D(const Coord3D& pos, const Coord3D& dir)
	{
		origin = pos;
		direction = dir;
	}

	template<typename Real>
	DYN_FUNC TRay3D<Real>::TRay3D(const TRay3D<Real>& ray)
	{
		origin = ray.origin;
		direction = ray.direction;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TRay3D<Real>::proximity(const TRay3D<Real>& ray) const
	{
		Coord3D u = origin - ray.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(ray.direction);
		Real c = ray.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(ray.direction);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			TPoint3D<Real> p0(origin);
			TPoint3D<Real> p1(ray.origin);

			TPoint3D<Real> q0 = p0.project(ray);
			TPoint3D<Real> q1 = p1.project(*this);

			return (q0 - p0).lengthSquared() < (q1 - p1).lengthSquared() ? q0 - p0 : p1 - q1;
		}

		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		if (sNum < 0) {
			sNum = 0;
			tNum = e;
			tDenom = c;
		}
		// Check t
		if (tNum < 0) {
			tNum = 0;
			if (-d < 0) {
				sNum = 0;
			}
			else {
				sNum = -d;
				sDenom = a;
			}
		}
		// Parameters of nearest points on restricted domain
		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return TSegment3D<Real>(origin + (s * direction), ray.origin + (t * direction));
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TRay3D<Real>::proximity(const TSegment3D<Real>& segment) const
	{
		Coord3D u = origin - segment.startPoint();
		Real a = direction.normSquared();
		Real b = direction.dot(segment.direction());
		Real c = segment.lengthSquared();
		Real d = u.dot(direction);
		Real e = u.dot(segment.direction());
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			if (a < REAL_EPSILON_SQUARED)
			{
				TPoint3D<Real> p0(origin);
				return p0.project(segment) - p0;
			}
			else
			{
				TPoint3D<Real> p1(segment.startPoint());
				TPoint3D<Real> p2(segment.endPoint());

				TPoint3D<Real> q1 = p1.project(*this);
				TPoint3D<Real> q2 = p2.project(*this);

				return (p1 - q1).lengthSquared() < (p2 - q2).lengthSquared() ? (p1 - q1) : (p2 - q2);
			}
		}

		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		if (sNum < 0) {
			sNum = 0;
			tNum = e;
			tDenom = c;
		}

		// Check t
		if (tNum < 0) {
			tNum = 0;
			if (-d < 0) {
				sNum = 0;
			}
			else {
				sNum = -d;
				sDenom = a;
			}
		}
		else if (tNum > tDenom) {
			tNum = tDenom;
			if ((-d + b) < 0) {
				sNum = 0;
			}
			else {
				sNum = -d + b;
				sDenom = a;
			}
		}

		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return TSegment3D<Real>(origin + (s * direction), segment.startPoint() + (t * segment.direction()));
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TRay3D<Real>::proximity(const TTriangle3D<Real>& triangle) const
	{
		TLine3D<Real> line(origin, direction);
		TSegment3D<Real> pq = line.proximity(triangle);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
		{
			return TPoint3D<Real>(origin).project(triangle) - TPoint3D<Real>(origin);
		}

		return pq;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TRay3D<Real>::proximity(const TRectangle3D<Real>& rectangle) const
	{
		TLine3D<Real> line(origin, direction);
		TSegment3D<Real> pq = line.proximity(rectangle);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
		{
			return TPoint3D<Real>(origin).project(rectangle) - TPoint3D<Real>(origin);
		}

		return pq;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TRay3D<Real>::proximity(const TAlignedBox3D<Real>& box) const
	{
		TLine3D<Real> line(origin, direction);
		TSegment3D<Real> pq = line.proximity(box);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
		{
			return TPoint3D<Real>(origin).project(box) - TPoint3D<Real>(origin);
		}

		return pq;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TRay3D<Real>::proximity(const TOrientedBox3D<Real>& obb) const
	{
		TLine3D<Real> line(origin, direction);
		TSegment3D<Real> pq = line.proximity(obb);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
		{
			return TPoint3D<Real>(origin).project(obb) - TPoint3D<Real>(origin);
		}

		return pq;
	}

	template<typename Real>
	DYN_FUNC Real TRay3D<Real>::distance(const TPoint3D<Real>& pt) const
	{
		return pt.distance(*this);
	}

	template<typename Real>
	DYN_FUNC Real TRay3D<Real>::distance(const TSegment3D<Real>& segment) const
	{
		return proximity(segment).length();
	}

	template<typename Real>
	DYN_FUNC Real TRay3D<Real>::distance(const TTriangle3D<Real>& triangle) const
	{
		return proximity(triangle).length();
	}

	template<typename Real>
	DYN_FUNC Real TRay3D<Real>::distanceSquared(const TPoint3D<Real>& pt) const
	{
		return pt.distanceSquared(*this);
	}

	template<typename Real>
	DYN_FUNC Real TRay3D<Real>::distanceSquared(const TSegment3D<Real>& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC Real TRay3D<Real>::distanceSquared(const TTriangle3D<Real>& triangle) const
	{
		return proximity(triangle).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC int TRay3D<Real>::intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const
	{
		Real DdN = direction.dot(plane.normal);
		if (abs(DdN) < REAL_EPSILON)
		{
			return 0;
		}

		Coord3D offset = origin - plane.origin;
		Real t = -offset.dot(plane.normal) / DdN;

		if (t < Real(0))
		{
			return 0;
		}

		interPt.origin = origin + t * direction;

		return 1;
	}

	template<typename Real>
	DYN_FUNC int TRay3D<Real>::intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const
	{
		Coord3D diff = origin - triangle.v[0];
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Coord3D normal = e0.cross(e1);

		Real DdN = direction.dot(normal);
		Real sign;
		if (DdN >= REAL_EPSILON)
		{
			sign = Real(1);
		}
		else if (DdN <= -REAL_EPSILON)
		{
			sign = (Real)-1;
			DdN = -DdN;
		}
		else
		{
			return 0;
		}

		Real DdQxE1 = sign * direction.dot(diff.cross(e1));
		if (DdQxE1 >= (Real)0)
		{
			Real DdE0xQ = sign * direction.dot(e0.cross(diff));
			if (DdE0xQ >= (Real)0)
			{
				if (DdQxE1 + DdE0xQ <= DdN)
				{
					// Line intersects triangle.
					Real QdN = -sign * diff.dot(normal);
					Real inv = (Real)1 / DdN;

					Real t = QdN * inv;

					if (t < Real(0))
					{
						return 0;
					}

					interPt.origin = origin + t * direction;
					return 1;
				}
				// else: b1+b2 > 1, no intersection
			}
			// else: b2 < 0, no intersection
		}
		// else: b1 < 0, no intersection

		return 0;
	}

	template<typename Real>
	DYN_FUNC int TRay3D<Real>::intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const
	{
		Coord3D diff = origin - sphere.center;
		Real a0 = diff.dot(diff) - sphere.radius * sphere.radius;
		Real a1 = direction.dot(diff);

		// Intersection occurs when Q(t) has real roots.
		Real discr = a1 * a1 - a0;
		if (discr > (Real)0)
		{
			Real root = glm::sqrt(discr);

			if (-a1 + root < Real(0))
			{
				return 0;
			}
			else if (-a1 + root < Real(0))
			{
				interSeg.startPoint() = origin + (-a1 + root) * direction;
				return 1;
			}
			else
			{
				interSeg.startPoint() = origin + (-a1 - root) * direction;
				interSeg.endPoint() = origin + (-a1 + root) * direction;
				return 2;
			}
		}
		else if (discr < Real(0))
		{
			return 0;
		}
		else
		{
			if (a1 > Real(0))
			{
				return 0;
			}
			interSeg.startPoint() = origin - a1 * direction;
			return 1;
		}
	}

	template<typename Real>
	DYN_FUNC int TRay3D<Real>::intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const
	{
		int interNum = TLine3D<Real>(origin, direction).intersect(abox, interSeg);
		if (interNum == 0)
		{
			return 0;
		}

		Real t0 = parameter(interSeg.startPoint());
		Real t1 = parameter(interSeg.endPoint());

		Interval<Real> iRay(0, REAL_MAX, false, true);

		if (iRay.inside(t0))
		{
			interSeg.v0 = origin + iRay.leftLimit() * direction;
			interSeg.v1 = origin + iRay.rightLimit() * direction;
			return 2;
		}
		else if (iRay.inside(t1))
		{
			interSeg.v0 = origin + iRay.leftLimit() * direction;
			interSeg.v1 = interSeg.v0;
			return 1;
		}
		else
		{
			return 0;
		}
	}

	template<typename Real>
	DYN_FUNC Real TRay3D<Real>::parameter(const Coord3D& pos) const
	{
		Coord3D l = pos - origin;
		Real d2 = direction.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(direction) / d2;
	}

	template<typename Real>
	DYN_FUNC bool TRay3D<Real>::isValid() const
	{
		return direction.normSquared() > REAL_EPSILON_SQUARED;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real>::TSegment3D()
	{
		v0 = Coord3D(0, 0, 0);
		v1 = Coord3D(1, 0, 0);
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real>::TSegment3D(const Coord3D& p0, const Coord3D& p1)
	{
		v0 = p0;
		v1 = p1;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real>::TSegment3D(const TSegment3D<Real>& segment)
	{
		v0 = segment.v0;
		v1 = segment.v1;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TSegment3D<Real>::proximity(const TSegment3D<Real>& segment) const
	{
		Coord3D u = v0 - segment.v0;
		Coord3D dir0 = v1 - v0;
		Coord3D dir1 = segment.v1 - segment.v0;
		Real a = dir0.normSquared();
		Real b = dir0.dot(dir1);
		Real c = dir1.normSquared();
		Real d = u.dot(dir0);
		Real e = u.dot(dir1);
		Real det = a * c - b * b;

		// Check for (near) parallelism
		if (det < REAL_EPSILON) {
			// Arbitrary choice
			Real l0 = lengthSquared();
			Real l1 = segment.lengthSquared();
			TPoint3D<Real> p0 = l0 < l1 ? TPoint3D<Real>(v0) : TPoint3D<Real>(segment.v0);
			TPoint3D<Real> p1 = l0 < l1 ? TPoint3D<Real>(v1) : TPoint3D<Real>(segment.v1);
			TSegment3D<Real> longerSeg = l0 < l1 ? segment : *this;
			bool bOpposite = l0 < l1 ? false : true;
			TPoint3D<Real> q0 = p0.project(longerSeg);
			TPoint3D<Real> q1 = p1.project(longerSeg);
			TSegment3D<Real> ret = p0.distance(q0) < p1.distance(q1) ? (q0 - p0) : (q1 - p1);
			return bOpposite ? -ret : ret;
		}

		// Find parameter values of closest points
		// on each segment��s infinite line. Denominator
		// assumed at this point to be ����det����,
		// which is always positive. We can check
		// value of numerators to see if we��re outside
		// the [0, 1] x [0, 1] domain.
		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		if (sNum < 0) {
			sNum = 0;
			tNum = e;
			tDenom = c;
		}
		else if (sNum > det) {
			sNum = det;
			tNum = e + b;
			tDenom = c;
		}

		// Check t
		if (tNum < 0) {
			tNum = 0;
			if (-d < 0) {
				sNum = 0;
			}
			else if (-d > a) {
				sNum = sDenom;
			}
			else {
				sNum = -d;
				sDenom = a;
			}
		}
		else if (tNum > tDenom) {
			tNum = tDenom;
			if ((-d + b) < 0) {
				sNum = 0;
			}
			else if ((-d + b) > a) {
				sNum = sDenom;
			}
			else {
				sNum = -d + b;
				sDenom = a;
			}
		}

		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return TSegment3D<Real>(v0 + (s * dir0), segment.v0 + (t * dir1));
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TSegment3D<Real>::proximity(const TPlane3D<Real>& plane) const
	{
		TPoint3D<Real> p0(v0);
		TPoint3D<Real> p1(v1);
		TPoint3D<Real> q0 = p0.project(plane);
		TPoint3D<Real> q1 = p1.project(plane);

		TSegment3D<Real> pq0 = q0 - p0;
		TSegment3D<Real> pq1 = q1 - p1;

		return pq0.lengthSquared() < pq1.lengthSquared() ? pq0 : pq1;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TSegment3D<Real>::proximity(const TTriangle3D<Real>& triangle) const
	{
		TLine3D<Real> line(startPoint(), direction());
		TSegment3D<Real> pq = line.proximity(triangle);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
			return TPoint3D<Real>(startPoint()).project(triangle) - TPoint3D<Real>(startPoint());

		if (t > Real(1))
			return TPoint3D<Real>(endPoint()).project(triangle) - TPoint3D<Real>(endPoint());

		return pq;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TSegment3D<Real>::proximity(const TRectangle3D<Real>& rectangle) const
	{
		TLine3D<Real> line(startPoint(), direction());
		TSegment3D<Real> pq = line.proximity(rectangle);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
			return TPoint3D<Real>(startPoint()).project(rectangle) - TPoint3D<Real>(startPoint());

		if (t > Real(1))
			return TPoint3D<Real>(endPoint()).project(rectangle) - TPoint3D<Real>(endPoint());

		return pq;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TSegment3D<Real>::proximity(const TAlignedBox3D<Real>& box) const
	{
		TLine3D<Real> line(startPoint(), direction());
		TSegment3D<Real> pq = line.proximity(box);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
			return TPoint3D<Real>(startPoint()).project(box) - TPoint3D<Real>(startPoint());

		if (t > Real(1))
			return TPoint3D<Real>(endPoint()).project(box) - TPoint3D<Real>(endPoint());

		return pq;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TSegment3D<Real>::proximity(const TOrientedBox3D<Real>& obb) const
	{
		TLine3D<Real> line(startPoint(), direction());
		TSegment3D<Real> pq = line.proximity(obb);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
			return TPoint3D<Real>(startPoint()).project(obb) - TPoint3D<Real>(startPoint());

		if (t > Real(1))
			return TPoint3D<Real>(endPoint()).project(obb) - TPoint3D<Real>(endPoint());

		return pq;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TSegment3D<Real>::proximity(const TTet3D<Real>& tet) const
	{
		Segment3D pq = proximity(tet.face(0));

		for (int i = 1; i < 4; i++)
		{
			Segment3D tmp = proximity(tet.face(i));
			if (tmp.length() < pq.length())
				pq = tmp;
		}
		return pq;
	}

	template<typename Real>
	DYN_FUNC Real TSegment3D<Real>::distance(const TSegment3D<Real>& segment) const
	{
		return proximity(segment).length();
	}

	template<typename Real>
	DYN_FUNC Real TSegment3D<Real>::distance(const TTriangle3D<Real>& triangle) const
	{
		return proximity(triangle).length();
	}

	template<typename Real>
	DYN_FUNC Real TSegment3D<Real>::distanceSquared(const TSegment3D<Real>& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC Real TSegment3D<Real>::distanceSquared(const TTriangle3D<Real>& triangle) const
	{
		return proximity(triangle).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC bool TSegment3D<Real>::intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const
	{
		Coord3D dir = direction();
		Real DdN = dir.dot(plane.normal);
		if (abs(DdN) < REAL_EPSILON)
		{
			return false;
		}

		Coord3D offset = v0 - plane.origin;
		Real t = -offset.dot(plane.normal) / DdN;

		if (t < Real(0) || t > Real(1))
		{
			return false;
		}

		interPt.origin = v0 + t * dir;
		return true;
	}

	template<typename Real>
	DYN_FUNC bool TSegment3D<Real>::intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const
	{
		Coord3D dir = direction();
		Coord3D diff = v0 - triangle.v[0];
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Coord3D normal = e0.cross(e1);

		Real DdN = dir.dot(normal);
		Real sign;
		if (DdN >= REAL_EPSILON)
		{
			sign = Real(1);
		}
		else if (DdN <= -REAL_EPSILON)
		{
			sign = (Real)-1;
			DdN = -DdN;
		}
		else
		{
			return false;
		}

		Real DdQxE1 = sign * dir.dot(diff.cross(e1));
		if (DdQxE1 >= (Real)0)
		{
			Real DdE0xQ = sign * dir.dot(e0.cross(diff));
			if (DdE0xQ >= (Real)0)
			{
				if (DdQxE1 + DdE0xQ <= DdN)
				{
					// Line intersects triangle.
					Real QdN = -sign * diff.dot(normal);
					Real inv = (Real)1 / DdN;

					Real t = QdN * inv;

					if (t < Real(0) || t > Real(1))
					{
						return false;
					}

					interPt.origin = v0 + t * dir;
					return true;
				}
				// else: b1+b2 > 1, no intersection
			}
			// else: b2 < 0, no intersection
		}
		// else: b1 < 0, no intersection

		return false;
	}

	template<typename Real>
	DYN_FUNC int TSegment3D<Real>::intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const
	{
		Coord3D diff = v0 - sphere.center;
		Coord3D dir = direction();
		Real a0 = diff.dot(diff) - sphere.radius * sphere.radius;
		Real a1 = dir.dot(diff);

		// Intersection occurs when Q(t) has real roots.
		Real discr = a1 * a1 - a0;
		if (discr > (Real)0)
		{
			Real root = glm::sqrt(discr);
			Real t1 = maximum(-a1 - root, Real(0));
			Real t2 = minimum(-a1 + root, Real(1));
			if (t1 < t2)
			{
				interSeg.startPoint() = v0 + t1 * dir;
				interSeg.endPoint() = v0 + t2 * dir;
				return 2;
			}
			else if (t1 > t2)
			{
				return 0;
			}
			else
			{
				interSeg.startPoint() = v0 + t1 * dir;
				return 1;
			}
		}
		else if (discr < (Real)0)
		{
			return 0;
		}
		else
		{
			Real t = -a1;
			if (t >= Real(0) && t <= Real(1))
			{
				interSeg.startPoint() = v0 - a1 * dir;
				return 1;
			}
			return 0;
		}
	}

	template<typename Real>
	DYN_FUNC int TSegment3D<Real>::intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const
	{
		Coord3D lineDir = direction();
		int interNum = TLine3D<Real>(v0, lineDir).intersect(abox, interSeg);
		if (interNum == 0)
		{
			return 0;
		}

		Real t0 = parameter(interSeg.startPoint());
		Real t1 = parameter(interSeg.endPoint());

		//Interval<Real> iSeg(0, 1, false, false);

		if ((0.0f <= t0 && t0 <= 1.0f) && (0.0f <= t1 && t1 <= 1.0f))
		{
			interSeg.v0 = v0 + t0 * lineDir;
			interSeg.v1 = v0 + t1 * lineDir;
			return 2;
		}
		else if ((0.0f <= t1 && t1 <= 1.0f))
		{
			interSeg.v0 = v0 + t1 * lineDir;
			interSeg.v1 = interSeg.v0;
			return 1;
		}
		else if ((0.0f <= t0 && t0 <= 1.0f))
		{
			interSeg.v0 = v0 + t0 * lineDir;
			interSeg.v1 = interSeg.v0;
			return 1;
		}
		else
		{
			return 0;
		}
	}

	template<typename Real>
	DYN_FUNC int TSegment3D<Real>::intersect(const TOrientedBox3D<Real>& obb, TSegment3D<Real>& interSeg) const
	{
		//transform to the local coordinate system of obb
		Coord3D diff0 = v0 - obb.center;
		Coord3D diff1 = v1 - obb.center;
		Coord3D v0Prime = Coord3D(diff0.dot(obb.u), diff0.dot(obb.v), diff0.dot(obb.w));
		Coord3D v1Prime = Coord3D(diff1.dot(obb.u), diff1.dot(obb.v), diff1.dot(obb.w));

		TSegment3D<Real> tmp = TSegment3D<Real>(v0Prime, v1Prime);

		int ret = tmp.intersect(TAlignedBox3D<Real>(-obb.extent, obb.extent), interSeg);

		Coord3D pPrime = interSeg.startPoint();
		Coord3D qPrime = interSeg.endPoint();

		//transform back to the global coordinate system
		Coord3D p = pPrime[0] * obb.u + pPrime[1] * obb.v + pPrime[2] * obb.w + obb.center;
		Coord3D q = qPrime[0] * obb.u + qPrime[1] * obb.v + qPrime[2] * obb.w + obb.center;

		interSeg = TSegment3D<Real>(p, q);

		return ret;
	}

	template<typename Real>
	DYN_FUNC Real TSegment3D<Real>::length() const
	{
		return (v1 - v0).norm();
	}

	template<typename Real>
	DYN_FUNC Real TSegment3D<Real>::lengthSquared() const
	{
		return (v1 - v0).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TSegment3D<Real>::parameter(const Coord3D& pos) const
	{
		Coord3D l = pos - v0;
		Coord3D dir = direction();
		Real d2 = dir.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(dir) / d2;
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TSegment3D<Real>::operator-(void) const
	{
		TSegment3D<Real> seg;
		seg.v0 = v1;
		seg.v1 = v0;

		return seg;
	}

	template<typename Real>
	DYN_FUNC bool TSegment3D<Real>::isValid() const
	{
		return lengthSquared() >= REAL_EPSILON_SQUARED;
	}

	template<typename Real>
	DYN_FUNC TPlane3D<Real>::TPlane3D()
	{
		origin = Coord3D(0);
		normal = Coord3D(0, 1, 0);
	}

	template<typename Real>
	DYN_FUNC TPlane3D<Real>::TPlane3D(const Coord3D& pos, const Coord3D n)
	{
		origin = pos;
		normal = n;
	}

	template<typename Real>
	DYN_FUNC TPlane3D<Real>::TPlane3D(const TPlane3D& plane)
	{
		origin = plane.origin;
		normal = plane.normal;
	}

	template<typename Real>
	DYN_FUNC bool TPlane3D<Real>::isValid() const
	{
		return normal.normSquared() >= REAL_EPSILON;
	}

	template<typename Real>
	DYN_FUNC TTriangle3D<Real>::TTriangle3D()
	{
		v[0] = Coord3D(0);
		v[1] = Coord3D(1, 0, 0);
		v[2] = Coord3D(0, 0, 1);
	}

	template<typename Real>
	DYN_FUNC TTriangle3D<Real>::TTriangle3D(const Coord3D& p0, const Coord3D& p1, const Coord3D& p2)
	{
		v[0] = p0;
		v[1] = p1;
		v[2] = p2;
	}

	template<typename Real>
	DYN_FUNC TTriangle3D<Real>::TTriangle3D(const TTriangle3D& triangle)
	{
		v[0] = triangle.v[0];
		v[1] = triangle.v[1];
		v[2] = triangle.v[2];
	}

	template<typename Real>
	DYN_FUNC Real TTriangle3D<Real>::area() const
	{
		return Real(0.5) * ((v[1] - v[0]).cross(v[2] - v[0])).norm();
	}

	template<typename Real>
	DYN_FUNC Coord3D TTriangle3D<Real>::normal() const
	{
		Coord3D n = (v[1] - v[0]).cross(v[2] - v[0]);
		if (n.norm() > REAL_EPSILON_SQUARED)
		{
			n.normalize();
		}
		return n;
		//return ((v[1] - v[0]) / (v[1] - v[0]).norm()).cross((v[2] - v[0]) / ((v[2] - v[0])).norm());
	}

	template<typename Real>
	DYN_FUNC bool TTriangle3D<Real>::computeBarycentrics(const Coord3D& p, Param& bary) const
	{
		if (!isValid())
		{
			bary.u = (Real)0;
			bary.v = (Real)0;
			bary.w = (Real)0;

			return false;
		}

		Coord3D q = TPoint3D<Real>(p).project(TPlane3D<Real>(v[0], normal())).origin;

		Coord3D d0 = v[1] - v[0];
		Coord3D d1 = v[2] - v[0];
		Coord3D d2 = q - v[0];
		Real d00 = d0.dot(d0);
		Real d01 = d0.dot(d1);
		Real d11 = d1.dot(d1);
		Real d20 = d2.dot(d0);
		Real d21 = d2.dot(d1);
		Real denom = d00 * d11 - d01 * d01;

		bary.v = (d11 * d20 - d01 * d21) / denom;
		bary.w = (d00 * d21 - d01 * d20) / denom;
		bary.u = Real(1) - bary.v - bary.w;

		return true;
	}


	template<typename Real>
	DYN_FUNC Coord3D TTriangle3D<Real>::computeLocation(const Param& bary) const
	{
		Coord3D d0 = v[1] - v[0];
		Coord3D d1 = v[2] - v[0];
		return v[0] + bary.v * d0 + bary.w * d1;
	}


	template<typename Real>
	DYN_FUNC Real TTriangle3D<Real>::maximumEdgeLength() const
	{
		return maximum(maximum((v[1] - v[0]).norm(), (v[2] - v[0]).norm()), (v[2] - v[1]).norm());
	}

	template<typename Real>
	DYN_FUNC bool TTriangle3D<Real>::isValid() const
	{
		return abs(area()) > REAL_EPSILON_SQUARED;
	}


	template<typename Real>
	DYN_FUNC TAlignedBox3D<Real> TTriangle3D<Real>::aabb()
	{
		TAlignedBox3D<Real> abox;
		abox.v0 = minimum(v[0], minimum(v[1], v[2]));
		abox.v1 = maximum(v[0], maximum(v[1], v[2]));

		return abox;
	}


	template<typename Real>
	DYN_FUNC Real TTriangle3D<Real>::distanceSquared(const TTriangle3D<Real>& triangle) const
	{
		Vector<Real, 3> p[3];
		p[0] = v[0];
		p[1] = v[1];
		p[2] = v[2];

		Vector<Real, 3> q[3];
		q[0] = triangle.v[0];
		q[1] = triangle.v[1];
		q[2] = triangle.v[2];

		//VF
		Real minD = (q[0] - p[0]).normSquared();

		//EF
		for (int st = 0; st < 3; st++)
		{
			int ind0 = st;
			int ind1 = (st + 1) % 3;
			minD = minimum(minD, TSegment3D<Real>(p[ind0], p[ind1]).distanceSquared(triangle));
		}

		for (int st = 0; st < 3; st++)
		{
			int ind0 = st;
			int ind1 = (st + 1) % 3;
			minD = minimum(minD, TSegment3D<Real>(q[ind0], q[ind1]).distanceSquared(*this));
		}

// 		for (int st = 0; st < 3; st++)
// 		{
// 			int ind0 = st;
// 			int ind1 = (st + 1) % 3;
// 			for (int ss = 0; ss < 3; ss++)
// 			{
// 				int ind2 = st;
// 				int ind3 = (st + 1) % 3;
// 
// 				minD = minimum(minD, TSegment3D<Real>(p[ind0], p[ind1]).distanceSquared(TSegment3D<Real>(p[ind2], p[ind3])));
// 			}
// 		}

		return minD;
	}

	template<typename Real>
	DYN_FUNC Real TTriangle3D<Real>::distance(const TTriangle3D<Real>& triangle) const
	{
		return glm::sqrt(distanceSquared(triangle));
	}

	template<typename Real>
	DYN_FUNC TRectangle3D<Real>::TRectangle3D()
	{
		center = Coord3D(0);
		axis[0] = Coord3D(1, 0, 0);
		axis[1] = Coord3D(0, 0, 1);
		extent = Coord2D(1, 1);
	}

	template<typename Real>
	DYN_FUNC TRectangle3D<Real>::TRectangle3D(const Coord3D& c, const Coord3D& a0, const Coord3D& a1, const Coord2D& ext)
	{
		center = c;
		axis[0] = a0;
		axis[1] = a1;
		extent = ext.maximum(Coord2D(0));
	}

	template<typename Real>
	DYN_FUNC TRectangle3D<Real>::TRectangle3D(const TRectangle3D<Real>& rectangle)
	{
		center = rectangle.center;
		axis[0] = rectangle.axis[0];
		axis[1] = rectangle.axis[1];
		extent = rectangle.extent;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TRectangle3D<Real>::vertex(const int i) const
	{
		int id = i % 4;
		switch (id)
		{
		case 0:
			return TPoint3D<Real>(center - axis[0] - axis[1]);
			break;
		case 1:
			return TPoint3D<Real>(center + axis[0] - axis[1]);
			break;
		case 2:
			return TPoint3D<Real>(center + axis[0] + axis[1]);
			break;
		default:
			return TPoint3D<Real>(center - axis[0] + axis[1]);
			break;
		}
	}

	template<typename Real>
	DYN_FUNC TSegment3D<Real> TRectangle3D<Real>::edge(const int i) const
	{
		return vertex(i + 1) - vertex(i);
	}

	template<typename Real>
	DYN_FUNC Real TRectangle3D<Real>::area() const
	{
		return Real(4) * extent[0] * extent[1];
	}

	template<typename Real>
	DYN_FUNC Coord3D TRectangle3D<Real>::normal() const
	{
		return axis[0].cross(axis[1]);
	}

	template<typename Real>
	DYN_FUNC bool TRectangle3D<Real>::computeParams(const Coord3D& p, Param& par) const
	{
		if (!isValid())
		{
			par.u = (Real)0;
			par.v = (Real)0;

			return false;
		}

		Coord3D offset = p - center;
		par.u = offset.dot(axis[0]);
		par.v = offset.dot(axis[1]);

		return true;
	}

	template<typename Real>
	DYN_FUNC bool TRectangle3D<Real>::isValid() const
	{
		bool bValid = true;
		bValid &= extent[0] >= REAL_EPSILON;
		bValid &= extent[1] >= REAL_EPSILON;

		bValid &= abs(axis[0].normSquared() - Real(1)) < REAL_EPSILON;
		bValid &= abs(axis[1].normSquared() - Real(1)) < REAL_EPSILON;

		bValid &= abs(axis[0].dot(axis[1])) < REAL_EPSILON;

		return bValid;
	}

	template<typename Real>
	DYN_FUNC TDisk3D<Real>::TDisk3D()
	{
		center = Coord3D(0);
		normal = Coord3D(0, 1, 0);
		radius = 1;
	}

	template<typename Real>
	DYN_FUNC TDisk3D<Real>::TDisk3D(const Coord3D& c, const Coord3D& n, const Real& r)
	{
		center = c;
		normal = n.norm() < REAL_EPSILON ? Coord3D(1, 0, 0) : n;
		normal.normalize();
		radius = r < 0 ? 0 : r;
	}

	template<typename Real>
	DYN_FUNC TDisk3D<Real>::TDisk3D(const TDisk3D& circle)
	{
		center = circle.center;
		normal = circle.normal;
		radius = circle.radius;
	}

	template<typename Real>
	DYN_FUNC Real TDisk3D<Real>::area()
	{
		return Real(M_PI) * radius * radius;
	}

	template<typename Real>
	DYN_FUNC bool TDisk3D<Real>::isValid()
	{
		return radius > REAL_EPSILON && normal.norm() > REAL_EPSILON;
	}

	template<typename Real>
	DYN_FUNC TSphere3D<Real>::TSphere3D()
	{
		center = Coord3D(0);
		radius = 1;
	}

	template<typename Real>
	DYN_FUNC TSphere3D<Real>::TSphere3D(const Coord3D& c, const Real& r)
	{
		center = c;
		radius = r < 0 ? 0 : r;
	}

	template<typename Real>
	DYN_FUNC TSphere3D<Real>::TSphere3D(const TSphere3D& sphere)
	{
		center = sphere.center;
		radius = sphere.radius;
	}

	template<typename Real>
	DYN_FUNC Real TSphere3D<Real>::volume()
	{
		return 4 * Real(M_PI) * radius * radius * radius / 3;
	}

	template<typename Real>
	DYN_FUNC bool TSphere3D<Real>::isValid()
	{
		return radius >= REAL_EPSILON;
	}


	template<typename Real>
	DYN_FUNC TAlignedBox3D<Real> TSphere3D<Real>::aabb()
	{
		TAlignedBox3D<Real> abox;
		abox.v0 = center - radius;
		abox.v1 = center + radius;

		return abox;
	}


	template<typename Real>
	DYN_FUNC TCapsule3D<Real>::TCapsule3D()
	{
		segment = TSegment3D<Real>(Coord3D(0), Coord3D(1, 0, 0));
		radius = Real(1);
	}

	template<typename Real>
	DYN_FUNC TCapsule3D<Real>::TCapsule3D(const Coord3D& v0, const Coord3D& v1, const Real& r)
		: segment(v0, v1)
		, radius(r)
	{

	}

	template<typename Real>
	DYN_FUNC TCapsule3D<Real>::TCapsule3D(const TCapsule3D& capsule)
	{
		segment = capsule.segment;
		radius = capsule.radius;
	}

	template<typename Real>
	DYN_FUNC Real TCapsule3D<Real>::volume()
	{
		Real r2 = radius * radius;
		return Real(M_PI) * r2 * segment.length() + Real(4.0 * M_PI / 3.0) * r2 * radius;
	}

	template<typename Real>
	DYN_FUNC bool TCapsule3D<Real>::isValid()
	{
		return radius >= REAL_EPSILON;
	}

	template<typename Real>
	DYN_FUNC TAlignedBox3D<Real> TCapsule3D<Real>::aabb()
	{
		TAlignedBox3D<Real> abox;
		abox.v0 = minimum(segment.v0, segment.v1) - radius;
		abox.v1 = maximum(segment.v0, segment.v1) + radius;
		return abox;
	}

	template<typename Real>
	DYN_FUNC TTet3D<Real>::TTet3D()
	{
		v[0] = Coord3D(0);
		v[1] = Coord3D(1, 0, 0);
		v[2] = Coord3D(0, 0, 1);
		v[3] = Coord3D(0, 1, 0);
	}

	template<typename Real>
	DYN_FUNC TTet3D<Real>::TTet3D(const Coord3D& v0, const Coord3D& v1, const Coord3D& v2, const Coord3D& v3)
	{
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
	}

	template<typename Real>
	DYN_FUNC TTet3D<Real>::TTet3D(const TTet3D& tet)
	{
		v[0] = tet.v[0];
		v[1] = tet.v[1];
		v[2] = tet.v[2];
		v[3] = tet.v[3];
	}

	template<typename Real>
	DYN_FUNC TTriangle3D<Real> TTet3D<Real>::face(const int index) const
	{
		switch (index)
		{
		case 0:
			return TTriangle3D<Real>(v[1], v[3], v[2]);
			break;
		case 1:
			return TTriangle3D<Real>(v[0], v[2], v[3]);
			break;
		case 2:
			return TTriangle3D<Real>(v[0], v[3], v[1]);
			break;
		case 3:
			return TTriangle3D<Real>(v[0], v[1], v[2]);
			break;
		default:
			break;
		}

		//return an ill triangle in case index is out of range
		return TTriangle3D<Real>(Coord3D(0), Coord3D(0), Coord3D(0));
	}

	template<typename Real>
	DYN_FUNC Real TTet3D<Real>::volume() const
	{
		Matrix3D M;
		M.setRow(0, v[1] - v[0]);
		M.setRow(1, v[2] - v[0]);
		M.setRow(2, v[3] - v[0]);

		return M.determinant() / Real(6);
	}

	template<typename Real>
	DYN_FUNC bool TTet3D<Real>::isValid()
	{
		return volume() >= REAL_EPSILON;
	}


	template<typename Real>
	DYN_FUNC void TTet3D<Real>::expand(Real r)
	{
		Coord3D center;
		center = (v[0] + v[1] + v[2] + v[3]) / 4.0f;
		for (int idx = 0; idx < 4; idx++)
		{
			Coord3D dir = (v[idx] - center);
			Real rr = dir.norm();
			dir /= dir.norm();
			rr += r;
			v[idx] = dir * rr + center;
		}
	}

	template<typename Real>
	DYN_FUNC bool TTet3D<Real>::intersect(const TTet3D<Real>& tet2, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance) const
	{


		//	printf("VVVVVVVVVVVVVVVVVVVVVVv\n");

		Coord3D mid = (tet2.v[0] + tet2.v[1] + tet2.v[2] + tet2.v[3]) / 4.0f;
		Coord3D v11 = mid + (tet2.v[0] - mid) / (tet2.v[0] - mid).norm() * ((tet2.v[0] - mid).norm());
		Coord3D v22 = mid + (tet2.v[1] - mid) / (tet2.v[1] - mid).norm() * ((tet2.v[1] - mid).norm());
		Coord3D v33 = mid + (tet2.v[2] - mid) / (tet2.v[2] - mid).norm() * ((tet2.v[2] - mid).norm());
		Coord3D v44 = mid + (tet2.v[3] - mid) / (tet2.v[3] - mid).norm() * ((tet2.v[3] - mid).norm());
		TTet3D<Real> tet(v11, v22, v33, v44);

		Real inter_dist_0 = 0.0f;
		Coord3D p11, p22, interNorm_0;

		Segment3D s[18];
		s[0] = Segment3D(v[0], v[1]);
		s[1] = Segment3D(v[0], v[2]);
		s[2] = Segment3D(v[0], v[3]);
		s[3] = Segment3D(v[1], v[2]);
		s[4] = Segment3D(v[1], v[3]);
		s[5] = Segment3D(v[2], v[3]);

		s[6] = Segment3D(v[0], (v[1] + v[2]) / 2.0f);
		s[7] = Segment3D(v[0], (v[1] + v[3]) / 2.0f);
		s[8] = Segment3D(v[0], (v[3] + v[2]) / 2.0f);
		s[9] = Segment3D(v[1], (v[3] + v[2]) / 2.0f);
		s[10] = Segment3D(v[1], (v[0] + v[2]) / 2.0f);
		s[11] = Segment3D(v[1], (v[0] + v[3]) / 2.0f);

		s[12] = Segment3D(v[2], (v[1] + v[0]) / 2.0f);
		s[13] = Segment3D(v[2], (v[1] + v[3]) / 2.0f);
		s[14] = Segment3D(v[2], (v[0] + v[3]) / 2.0f);
		s[15] = Segment3D(v[3], (v[1] + v[2]) / 2.0f);
		s[16] = Segment3D(v[3], (v[0] + v[2]) / 2.0f);
		s[17] = Segment3D(v[3], (v[1] + v[0]) / 2.0f);



		for (int i = 0; i < 18; i++)
		{
			if (i >= 6 && !(inter_dist_0 < -0.000f))
				break;
			Line3D l_i = Line3D(s[i].v0, s[i].direction());
			Segment3D tmp_seg;

			if (l_i.intersect(tet, tmp_seg))
			{

				Real left = (tmp_seg.v0 - s[i].v0).dot(s[i].direction().normalize());
				Real right = (tmp_seg.v1 - s[i].v0).dot(s[i].direction().normalize());
				if (right < left)
				{
					Real tmp = left;
					left = right;
					right = tmp;
				}
				Real maxx = (s[i].v1 - s[i].v0).dot(s[i].direction().normalize());
				if (right < 0 || left > maxx)
					continue;
				left = maximum(left, 0.0f);
				right = minimum(right, maxx);

				p1 = s[i].v0 + ((left + right) / 2.0f * s[i].direction().normalize());
				Bool tmp_bool;
				p2 = Point3D(p1).project(tet, tmp_bool).origin;//
				interDist = -(p1 - p2).norm();

				if (need_distance)
				{
					Coord3D p11 = s[i].v0 + left * s[i].direction().normalize();
					Coord3D p22 = Point3D(p11).project(tet, tmp_bool).origin;
					if ((p11 - p22).norm() > abs(interDist))
					{
						p1 = p11; p2 = p22;
						interDist = -(p1 - p2).norm();
					}

					p11 = s[i].v0 + right * s[i].direction().normalize();
					p22 = Point3D(p11).project(tet, tmp_bool).origin;
					if ((p11 - p22).norm() > abs(interDist))
					{
						p1 = p11; p2 = p22;
						interDist = -(p1 - p2).norm();
					}
				}
				Coord3D dir = (p1 - p2) / interDist;
				interNorm = dir;
				//dir *= -1;

				if (interDist < inter_dist_0)
				{
					inter_dist_0 = interDist;
					p11 = p1;
					p22 = p2;
					interNorm_0 = interNorm;
				}

				if (need_distance == 0)
					if (inter_dist_0 < -0.000f)
					{
						interDist = inter_dist_0;
						p1 = p11;
						p2 = p22;
						interNorm = interNorm_0;
						interDist += 0.0000f;
						return true;
					}

			}
		}

		if (inter_dist_0 < -0.000f)
		{
			interDist = inter_dist_0;
			p1 = p11;
			p2 = p22;
			interNorm = interNorm_0;
			interDist += 0.0000f;
			return true;
		}
		return false;
	}


	template<typename Real>
	DYN_FUNC bool TTet3D<Real>::intersect(const TTriangle3D<Real>& tri, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance) const
	{


		//	printf("VVVVVVVVVVVVVVVVVVVVVVv\n");


		TTet3D<Real> tet(v[0], v[1], v[2], v[3]);

		Real inter_dist_0 = 0.0f;
		Coord3D p11, p22, interNorm_0;

		Segment3D s[18];

		s[0] = Segment3D(tri.v[0], tri.v[1]);
		s[1] = Segment3D(tri.v[0], tri.v[2]);
		s[2] = Segment3D(tri.v[0], tri.v[2]);

		for (int i = 0; i < 3; i++)
		{

			Line3D l_i = Line3D(s[i].v0, s[i].direction());
			Segment3D tmp_seg;

			if (l_i.intersect(tet, tmp_seg))
			{

				Real left = (tmp_seg.v0 - s[i].v0).dot(s[i].direction().normalize());
				Real right = (tmp_seg.v1 - s[i].v0).dot(s[i].direction().normalize());
				if (right < left)
				{
					Real tmp = left;
					left = right;
					right = tmp;
				}
				Real maxx = (s[i].v1 - s[i].v0).dot(s[i].direction().normalize());
				if (right < 0 || left > maxx)
					continue;
				left = maximum(left, 0.0f);
				right = minimum(right, maxx);

				p2 = s[i].v0 + ((left + right) / 2.0f * s[i].direction().normalize());
				Bool tmp_bool;
				p1 = Point3D(p2).project(tet, tmp_bool).origin;//
				interDist = -(p1 - p2).norm();


				Coord3D p11 = s[i].v0 + left * s[i].direction().normalize();
				Coord3D p22 = Point3D(p11).project(tet, tmp_bool).origin;
				if ((p11 - p22).norm() > abs(interDist))
				{
					p2 = p11; p1 = p22;
					interDist = -(p1 - p2).norm();
				}

				p11 = s[i].v0 + right * s[i].direction().normalize();
				p22 = Point3D(p11).project(tet, tmp_bool).origin;
				if ((p11 - p22).norm() > abs(interDist))
				{
					p2 = p11; p1 = p22;
					interDist = -(p1 - p2).norm();
				}

				Coord3D dir = (p1 - p2) / interDist;
				interNorm = dir;
				//dir *= -1;

				if (interDist < inter_dist_0)
				{
					inter_dist_0 = interDist;
					p11 = p1;
					p22 = p2;
					interNorm_0 = interNorm;
				}



			}
		}

		if (inter_dist_0 < -0.000f)
		{
			interDist = inter_dist_0;
			p1 = p11;
			p2 = p22;
			interNorm = interNorm_0;
			interDist += 0.0000f;
			return true;
		}

		s[0] = Segment3D(v[0], v[1]);
		s[1] = Segment3D(v[0], v[2]);
		s[2] = Segment3D(v[0], v[3]);
		s[3] = Segment3D(v[1], v[2]);
		s[4] = Segment3D(v[1], v[3]);
		s[5] = Segment3D(v[2], v[3]);

		s[6] = Segment3D(v[0], (v[1] + v[2]) / 2.0f);
		s[7] = Segment3D(v[0], (v[1] + v[3]) / 2.0f);
		s[8] = Segment3D(v[0], (v[3] + v[2]) / 2.0f);
		s[9] = Segment3D(v[1], (v[3] + v[2]) / 2.0f);
		s[10] = Segment3D(v[1], (v[0] + v[2]) / 2.0f);
		s[11] = Segment3D(v[1], (v[0] + v[3]) / 2.0f);

		s[12] = Segment3D(v[2], (v[1] + v[0]) / 2.0f);
		s[13] = Segment3D(v[2], (v[1] + v[3]) / 2.0f);
		s[14] = Segment3D(v[2], (v[0] + v[3]) / 2.0f);
		s[15] = Segment3D(v[3], (v[1] + v[2]) / 2.0f);
		s[16] = Segment3D(v[3], (v[0] + v[2]) / 2.0f);
		s[17] = Segment3D(v[3], (v[1] + v[0]) / 2.0f);



		for (int i = 0; i < 18; i++)
		{

			Line3D l_i = Line3D(s[i].v0, s[i].direction());
			Point3D tmp_point;

			if (l_i.intersect(tri, tmp_point))
			{

				Real left = (tmp_point.origin - s[i].v0).dot(s[i].direction().normalize());
				Real maxx = (s[i].v1 - s[i].v0).dot(s[i].direction().normalize());

				if (left < 0 || left > maxx)
					continue;

				if (left < maxx / 2.0f)
					p1 = s[i].v0;
				else
					p1 = s[i].v1;
				p2 = tmp_point.origin;//Point3D(p1).project(tet, tmp_bool).origin;
				interDist = -(p1 - p2).norm();
				Coord3D dir = (p1 - p2) / interDist;
				interNorm = dir;

			}
		}
		if (inter_dist_0 < -0.000f)
		{
			interDist = inter_dist_0;
			p1 = p11;
			p2 = p22;
			interNorm = interNorm_0;
			interDist += 0.0000f;
			return true;
		}




		return false;
	}
	template<typename Real>
	DYN_FUNC TAlignedBox3D<Real> TTet3D<Real>::aabb()
	{
		TAlignedBox3D<Real> abox;
		abox.v0[0] = minimum(minimum(v[0][0], v[1][0]), minimum(v[2][0], v[3][0]));
		abox.v0[1] = minimum(minimum(v[0][1], v[1][1]), minimum(v[2][1], v[3][1]));
		abox.v0[2] = minimum(minimum(v[0][2], v[1][2]), minimum(v[2][2], v[3][2]));

		abox.v1[0] = maximum(maximum(v[0][0], v[1][0]), maximum(v[2][0], v[3][0]));
		abox.v1[1] = maximum(maximum(v[0][1], v[1][1]), maximum(v[2][1], v[3][1]));
		abox.v1[2] = maximum(maximum(v[0][2], v[1][2]), maximum(v[2][2], v[3][2]));

		return abox;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TTet3D<Real>::circumcenter() const
	{
		// Use coordinates relative to point `a' of the tetrahedron.

		// ba = b - a
		Real ba_x = v[1][0] - v[0][0];
		Real ba_y = v[1][1] - v[0][1];
		Real ba_z = v[1][2] - v[0][2];

		// ca = c - a
		Real ca_x = v[2][0] - v[0][0];
		Real ca_y = v[2][1] - v[0][1];
		Real ca_z = v[2][2] - v[0][2];

		// da = d - a
		Real da_x = v[3][0] - v[0][0];
		Real da_y = v[3][1] - v[0][1];
		Real da_z = v[3][2] - v[0][2];

		// Squares of lengths of the edges incident to `a'.
		Real len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z;
		Real len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z;
		Real len_da = da_x * da_x + da_y * da_y + da_z * da_z;

		// Cross products of these edges.

		// c cross d
		Real cross_cd_x = ca_y * da_z - da_y * ca_z;
		Real cross_cd_y = ca_z * da_x - da_z * ca_x;
		Real cross_cd_z = ca_x * da_y - da_x * ca_y;

		// d cross b
		Real cross_db_x = da_y * ba_z - ba_y * da_z;
		Real cross_db_y = da_z * ba_x - ba_z * da_x;
		Real cross_db_z = da_x * ba_y - ba_x * da_y;

		// b cross c
		Real cross_bc_x = ba_y * ca_z - ca_y * ba_z;
		Real cross_bc_y = ba_z * ca_x - ca_z * ba_x;
		Real cross_bc_z = ba_x * ca_y - ca_x * ba_y;

		// Calculate the denominator of the formula.
		Real denominator = 0.5 / (ba_x * cross_cd_x + ba_y * cross_cd_y + ba_z * cross_cd_z);

		// Calculate offset (from `a') of circumcenter.
		Real circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x + len_da * cross_bc_x) * denominator;
		Real circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y + len_da * cross_bc_y) * denominator;
		Real circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z + len_da * cross_bc_z) * denominator;

		return TPoint3D<Real>(circ_x + v[0][0], circ_y + v[0][1], circ_z + v[0][2]);
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TTet3D<Real>::barycenter() const
	{
		return TPoint3D<Real>(Real(0.25) * (v[0] + v[1] + v[2] + v[3]));
	}

	template<typename Real>
	DYN_FUNC TAlignedBox3D<Real>::TAlignedBox3D()
	{
		v0 = Coord3D(Real(-1));
		v1 = Coord3D(Real(1));
	}

	template<typename Real>
	DYN_FUNC TAlignedBox3D<Real>::TAlignedBox3D(const Coord3D& p0, const Coord3D& p1)
	{
		v0 = p0;
		v1 = p1;
	}

	template<typename Real>
	DYN_FUNC TAlignedBox3D<Real>::TAlignedBox3D(const TAlignedBox3D& box)
	{
		v0 = box.v0;
		v1 = box.v1;
	}

	template<typename Real>
	DYN_FUNC bool TAlignedBox3D<Real>::intersect(const TAlignedBox3D& abox, TAlignedBox3D& interBox) const
	{
		for (int i = 0; i < 3; i++)
		{
			if (v1[i] <= abox.v0[i] || v0[i] >= abox.v1[i])
			{
				return false;
			}
		}

		interBox.v0 = v0.maximum(abox.v0);
		interBox.v1 = v1.minimum(abox.v1);

		for (int i = 0; i < 3; i++)
		{
			if (v1[i] <= abox.v1[i])
			{
				interBox.v1[i] = v1[i];
			}
			else
			{
				interBox.v1[i] = abox.v1[i];
			}

			if (v0[i] <= abox.v0[i])
			{
				interBox.v0[i] = abox.v0[i];
			}
			else
			{
				interBox.v0[i] = v0[i];
			}
		}

		return true;
	}

	template<typename Real>
	DYN_FUNC bool TAlignedBox3D<Real>::meshInsert(const TTriangle3D<Real>& tri) const
	{
		//return true;
		TPoint3D<Real> p0 = TPoint3D<Real>(tri.v[0]);
		TPoint3D<Real> p1 = TPoint3D<Real>(tri.v[1]);
		TPoint3D<Real> p2 = TPoint3D<Real>(tri.v[2]);
		TAlignedBox3D<Real> AABB = TAlignedBox3D<Real>(v0, v1);
		if (p0.inside(AABB))return true;
		if (p1.inside(AABB))return true;
		if (p2.inside(AABB))return true;

		TPoint3D<Real> pTmp = TPoint3D<Real>((v0 + v1) / 2);
		Real r = (pTmp.origin - v0).norm();
		//		if (abs(pTmp.distance(tri)) < r) return true;

		Coord3D P0 = v0;
		Coord3D P1 = Coord3D(v0[0], v0[1], v1[2]);
		Coord3D P2 = Coord3D(v0[0], v1[1], v0[2]);
		Coord3D P3 = Coord3D(v0[0], v1[1], v1[2]);
		Coord3D P4 = Coord3D(v1[0], v0[1], v0[2]);
		Coord3D P5 = Coord3D(v1[0], v0[1], v1[2]);
		Coord3D P6 = Coord3D(v1[0], v1[1], v0[2]);
		Coord3D P7 = v1;

		TPoint3D<Real> interpt;
		Coord3D seg[12][2] = {
			P0,P1,
			P0,P2,
			P0,P4,
			P7,P6,
			P7,P5,
			P7,P3,
			P2,P3,
			P2,P6,
			P3,P1,
			P6,P4,
			P4,P5,
			P1,P5
		};

		Coord3D m_triangle_index[12][3] = {
			P0,P1,P2,
			P0,P1,P4,
			P0,P2,P4,
			P7,P5,P6,
			P7,P3,P5,
			P7,P3,P6,
			P2,P4,P6,
			P1,P4,P5,
			P1,P2,P3,
			P3,P5,P1,
			P5,P6,P4,
			P3,P6,P2
		};

		for (int i = 0; i < 12; i++)
		{
			TTriangle3D<Real> t_tmp = TTriangle3D<Real>(m_triangle_index[i][0], m_triangle_index[i][1], m_triangle_index[i][2]);
			TSegment3D<Real> s_tmp = TSegment3D<Real>(seg[i][0], seg[i][1]);
			TSegment3D<Real> s0, s1, s2;
			s0 = TSegment3D<Real>(tri.v[0], tri.v[1]);
			s1 = TSegment3D<Real>(tri.v[1], tri.v[2]);
			s2 = TSegment3D<Real>(tri.v[2], tri.v[0]);

			if ((s0.intersect(t_tmp, interpt)) || (s1.intersect(t_tmp, interpt)) || (s2.intersect(t_tmp, interpt)) || (s_tmp.intersect(tri, interpt)))
				return true;
		}
		//printf("!!!!!!!!!!!!!!!!!!FALSE CASE:\n");

			/*
			\nAABB:\n%.3lf %.3lf %.3lf\nTRI:\n%.3lf %.3lf %.3lf\n%.3lf %.3lf %.3lf\n%.3lf %.3lf %.3lf\nPPPPPPPPPPPPPPP\n",
			v0[0], v0[1], v0[2],
			v1[0], v1[1], v1[2],
			tri.v[0][0], tri.v[0][1], tri.v[0][2],
			tri.v[1][0], tri.v[1][1], tri.v[1][2],
			tri.v[2][0], tri.v[2][1], tri.v[2][2]
			);*/
		return false;
	}

	template<typename Real>
	DYN_FUNC TOrientedBox3D<Real> TAlignedBox3D<Real>::rotate(const Matrix3D& mat)
	{
		TOrientedBox3D<Real> obox;
		obox.u = mat * Coord3D(1, 0, 0);
		obox.v = mat * Coord3D(0, 1, 0);
		obox.w = mat * Coord3D(0, 0, 1);
		obox.center = 0.5f * (v0 + v1);
		obox.extent = 0.5f * (v1 - v0);

		return obox;
	}


	template<typename Real>
	DYN_FUNC bool TAlignedBox3D<Real>::isValid()
	{
		return v1[0] > v0[0] && v1[1] > v0[1] && v1[2] > v0[2];
	}

	template<typename Real>
	DYN_FUNC TOrientedBox3D<Real>::TOrientedBox3D()
	{
		center = Coord3D(0);
		u = Coord3D(1, 0, 0);
		v = Coord3D(0, 1, 0);
		w = Coord3D(0, 0, 1);

		extent = Coord3D(1);
	}

	template<typename Real>
	DYN_FUNC TOrientedBox3D<Real>::TOrientedBox3D(const Coord3D c, const Coord3D u_t, const Coord3D v_t, const Coord3D w_t, const Coord3D ext)
	{
		center = c;
		u = u_t;
		v = v_t;
		w = w_t;
		extent = ext;
	}

	template<typename Real>
	DYN_FUNC TOrientedBox3D<Real>::TOrientedBox3D(const Coord3D c, const Quat<Real> rot, const Coord3D ext)
	{
		center = c;
		extent = ext;

		auto mat = rot.toMatrix3x3();
		u = mat.col(0);
		v = mat.col(1);
		w = mat.col(2);
	}

	template<typename Real>
	DYN_FUNC TOrientedBox3D<Real>::TOrientedBox3D(const TOrientedBox3D& obb)
	{
		center = obb.center;
		u = obb.u;
		v = obb.v;
		w = obb.w;
		extent = obb.extent;
	}


	template<typename Real>
	DYN_FUNC Real TOrientedBox3D<Real>::volume()
	{
		return 8 * extent[0] * extent[1] * extent[2];
	}

	template<typename Real>
	DYN_FUNC bool TOrientedBox3D<Real>::isValid()
	{
		bool bValid = true;
		bValid &= extent[0] >= REAL_EPSILON;
		bValid &= extent[1] >= REAL_EPSILON;
		bValid &= extent[2] >= REAL_EPSILON;

		bValid &= abs(u.normSquared() - Real(1)) < REAL_EPSILON;
		bValid &= abs(v.normSquared() - Real(1)) < REAL_EPSILON;
		bValid &= abs(w.normSquared() - Real(1)) < REAL_EPSILON;

		bValid &= abs(u.dot(v)) < REAL_EPSILON;
		bValid &= abs(v.dot(w)) < REAL_EPSILON;
		bValid &= abs(w.dot(u)) < REAL_EPSILON;

		return bValid;
	}


	template<typename Real>
	DYN_FUNC TOrientedBox3D<Real> TOrientedBox3D<Real>::rotate(const Matrix3D& mat)
	{
		TOrientedBox3D<Real> obox;
		obox.center = center;
		obox.extent = extent;
		obox.u = mat * obox.u;
		obox.v = mat * obox.v;
		obox.w = mat * obox.w;

		return obox;
	}


	template<typename Real>
	DYN_FUNC TAlignedBox3D<Real> TOrientedBox3D<Real>::aabb()
	{
		TAlignedBox3D<Real> abox;

		Coord3D r_ext;
		r_ext[0] = abs(extent[0] * u[0]) + abs(extent[1] * v[0]) + abs(extent[2] * w[0]);
		r_ext[1] = abs(extent[0] * u[1]) + abs(extent[1] * v[1]) + abs(extent[2] * w[1]);
		r_ext[2] = abs(extent[0] * u[2]) + abs(extent[1] * v[2]) + abs(extent[2] * w[2]);

		abox.v0 = center - r_ext;
		abox.v1 = center + r_ext;

		return abox;
	}



	template<typename Real>
	DYN_FUNC bool TOrientedBox3D<Real>::point_intersect(const TOrientedBox3D<Real>& OBB, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2) const
	{
		Point3D p[8];
		p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
		p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
		p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
		p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
		p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
		p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
		p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
		p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);

		Segment3D s[12];
		s[0] = Segment3D(p[0].origin, p[1].origin); s[1] = Segment3D(p[0].origin, p[2].origin); s[2] = Segment3D(p[0].origin, p[4].origin);
		s[3] = Segment3D(p[3].origin, p[1].origin); s[4] = Segment3D(p[3].origin, p[2].origin); s[5] = Segment3D(p[3].origin, p[7].origin);
		s[6] = Segment3D(p[6].origin, p[7].origin); s[7] = Segment3D(p[6].origin, p[2].origin); s[8] = Segment3D(p[6].origin, p[4].origin);
		s[9] = Segment3D(p[5].origin, p[7].origin); s[10] = Segment3D(p[5].origin, p[1].origin); s[11] = Segment3D(p[5].origin, p[4].origin);

		int min_idx1 = -1;
		int min_idx2 = -1;
		Real distance = 1.0f;

		Plane3D plane1, plane2;
		Coord3D axis;
		Real inter;

		//check u axis
		Real min_u, max_u;
		for (int i = 0; i < 8; i++)
		{
			Coord3D offset_self = p[i].origin - OBB.center;
			Real tmp = offset_self.dot(OBB.u);
			if (i == 0) min_u = max_u = tmp;
			else
			{
				min_u = minimum(min_u, tmp);
				max_u = maximum(max_u, tmp);
			}
		}
		if (max_u < -OBB.extent[0] || min_u > OBB.extent[0]) return false;
		Real inter_u = minimum(minimum(max_u + OBB.extent[0], OBB.extent[0] - min_u), minimum(2 * OBB.extent[0], max_u - min_u));
		axis = OBB.u;
		inter = inter_u;
		plane1 = Plane3D((OBB.center + OBB.u * OBB.extent[0]), OBB.u);
		plane2 = Plane3D((OBB.center - OBB.u * OBB.extent[0]), -OBB.u);

		//check v axis
		Real min_v, max_v;
		for (int i = 0; i < 8; i++)
		{
			Coord3D offset_self = p[i].origin - OBB.center;
			Real tmp = offset_self.dot(OBB.v);
			if (i == 0) min_v = max_v = tmp;
			else
			{
				min_v = minimum(min_v, tmp);
				max_v = maximum(max_v, tmp);
			}
		}
		if (max_v < -OBB.extent[1] || min_v > OBB.extent[1]) return false;
		Real inter_v = minimum(minimum(max_v + OBB.extent[1], OBB.extent[1] - min_v), minimum(2 * OBB.extent[1], max_v - min_v));
		if (inter_v < inter)
		{
			axis = OBB.v;
			inter = inter_v;
			plane1 = Plane3D((OBB.center + OBB.v * OBB.extent[1]), OBB.v);
			plane2 = Plane3D((OBB.center - OBB.v * OBB.extent[1]), -OBB.v);
		}

		//check w axis
		Real min_w, max_w;
		for (int i = 0; i < 8; i++)
		{
			Coord3D offset_self = p[i].origin - OBB.center;
			Real tmp = offset_self.dot(OBB.w);
			if (i == 0) min_w = max_w = tmp;
			else
			{
				min_w = minimum(min_w, tmp);
				max_w = maximum(max_w, tmp);
			}
		}
		if (max_w < -OBB.extent[2] || min_w > OBB.extent[2]) { return false; }
		Real inter_w = minimum(minimum(max_w + OBB.extent[2], OBB.extent[2] - min_w), minimum(2 * OBB.extent[2], max_w - min_w));
		if (inter_w < inter)
		{
			axis = OBB.w;
			inter = inter_w;
			plane1 = Plane3D((OBB.center + OBB.w * OBB.extent[2]), OBB.w);
			plane2 = Plane3D((OBB.center - OBB.w * OBB.extent[2]), -OBB.w);
		}

		//printf("$$$$$$$$$$$$$$$$$$$$ INSIDE CHECKPOINT\n");

		for (int i = 0; i < 8; i++)
		{
			Real tmp = -minimum(abs(p[i].distance(plane1)), abs(p[i].distance(plane2)));
			if (tmp < distance && p[i].distance(OBB) < EPSILON)
			{
				distance = tmp;
				min_idx1 = i;
			}
		}

		for (int i = 0; i < 12; i++)
		{
			Segment3D tmp;
			if (s[i].intersect(OBB, tmp))
			{
				Point3D inp((tmp.startPoint() + tmp.endPoint()) / 2.0f);
				Real tmp_distance = -minimum(abs(inp.distance(plane1)), abs(inp.distance(plane2)));
				if (tmp_distance < distance)
				{

					//printf(" ---------------------------------- interSegStart: %.3lf %.3lf %.3lf interSegEnd: %.3lf %.3lf %.3lf \n", 
					//	tmp.startPoint()[0], tmp.startPoint()[1], tmp.startPoint()[2],
					//	tmp.endPoint()[0], tmp.endPoint()[1], tmp.endPoint()[2]);
					distance = tmp_distance;
					min_idx2 = i;
					min_idx1 = -1;
				}

			}
		}
		//printf("distance = %.10lf\n", distance);
		if (distance > EPSILON) return false;
		interDist = distance;

		Point3D pt;
		if (min_idx1 != -1)
			pt = p[min_idx1];
		else
		{
			Segment3D tmp;
			s[min_idx2].intersect(OBB, tmp);
			pt = Point3D((tmp.startPoint() + tmp.endPoint()) / 2.0f);
		}


		p1 = pt.origin;

		if (abs(pt.distance(plane1)) < abs(pt.distance(plane2)))
		{
			p2 = pt.project(plane1).origin;
			interNorm = axis;
		}
		else
		{
			p2 = pt.project(plane2).origin;
			interNorm = -axis;
		}

		return true;
	}

	template<typename Real>
	DYN_FUNC bool TOrientedBox3D<Real>::point_intersect(const TTet3D<Real>& TET, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2) const
	{
		Point3D p[8];
		p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
		p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
		p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
		p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
		p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
		p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
		p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
		p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);

		Segment3D s[12];
		s[0] = Segment3D(p[0].origin, p[1].origin); s[1] = Segment3D(p[0].origin, p[2].origin); s[2] = Segment3D(p[0].origin, p[4].origin);
		s[3] = Segment3D(p[3].origin, p[1].origin); s[4] = Segment3D(p[3].origin, p[2].origin); s[5] = Segment3D(p[3].origin, p[7].origin);
		s[6] = Segment3D(p[6].origin, p[7].origin); s[7] = Segment3D(p[6].origin, p[2].origin); s[8] = Segment3D(p[6].origin, p[4].origin);
		s[9] = Segment3D(p[5].origin, p[7].origin); s[10] = Segment3D(p[5].origin, p[1].origin); s[11] = Segment3D(p[5].origin, p[4].origin);

		Segment3D s_tet[18];
		s_tet[0] = Segment3D(TET.v[0], TET.v[1]);
		s_tet[1] = Segment3D(TET.v[0], TET.v[2]);
		s_tet[2] = Segment3D(TET.v[0], TET.v[3]);
		s_tet[3] = Segment3D(TET.v[1], TET.v[2]);
		s_tet[4] = Segment3D(TET.v[1], TET.v[3]);
		s_tet[5] = Segment3D(TET.v[2], TET.v[3]);

		s_tet[6] = Segment3D(TET.v[0], (TET.v[1] + TET.v[2]) / 2.0f);
		s_tet[7] = Segment3D(TET.v[0], (TET.v[1] + TET.v[3]) / 2.0f);
		s_tet[8] = Segment3D(TET.v[0], (TET.v[3] + TET.v[2]) / 2.0f);
		s_tet[9] = Segment3D(TET.v[1], (TET.v[3] + TET.v[2]) / 2.0f);
		s_tet[10] = Segment3D(TET.v[1], (TET.v[0] + TET.v[2]) / 2.0f);
		s_tet[11] = Segment3D(TET.v[1], (TET.v[0] + TET.v[3]) / 2.0f);

		s_tet[12] = Segment3D(TET.v[2], (TET.v[1] + TET.v[0]) / 2.0f);
		s_tet[13] = Segment3D(TET.v[2], (TET.v[1] + TET.v[3]) / 2.0f);
		s_tet[14] = Segment3D(TET.v[2], (TET.v[0] + TET.v[3]) / 2.0f);
		s_tet[15] = Segment3D(TET.v[3], (TET.v[1] + TET.v[2]) / 2.0f);
		s_tet[16] = Segment3D(TET.v[3], (TET.v[0] + TET.v[2]) / 2.0f);
		s_tet[17] = Segment3D(TET.v[3], (TET.v[1] + TET.v[0]) / 2.0f);


		interDist = 0.0f;
		//obb intersect tet
		for (int i = 0; i < 12; i++)
		{
			Line3D l_i = Line3D(s[i].v0, s[i].direction());
			Segment3D tmp_seg;

			if (l_i.intersect(TET, tmp_seg))
			{

				Real left = (tmp_seg.v0 - s[i].v0).dot(s[i].direction().normalize());
				Real right = (tmp_seg.v1 - s[i].v0).dot(s[i].direction().normalize());
				if (right < left)
				{
					Real tmp = left;
					left = right;
					right = tmp;
				}
				Real maxx = (s[i].v1 - s[i].v0).dot(s[i].direction().normalize());
				if (right < 0 || left > maxx)
					continue;
				left = maximum(left, 0.0f);
				right = minimum(right, maxx);

				Bool tmp_bool;

				Coord3D p11 = s[i].v0 + ((left + right) / 2.0f * s[i].direction().normalize());
				Coord3D p22 = Point3D(p11).project(TET, tmp_bool).origin;//


				if ((p11 - p22).norm() > abs(interDist))
				{
					interDist = -(p11 - p22).norm();
					p1 = p11;
					p2 = p22;
				}
				p11 = s[i].v0 + left * s[i].direction().normalize();
				p22 = Point3D(p11).project(TET, tmp_bool).origin;
				if ((p11 - p22).norm() > abs(interDist))
				{
					p1 = p11; p2 = p22;
					interDist = -(p1 - p2).norm();
				}

				p11 = s[i].v0 + right * s[i].direction().normalize();
				p22 = Point3D(p11).project(TET, tmp_bool).origin;
				if ((p11 - p22).norm() > abs(interDist))
				{
					p1 = p11; p2 = p22;
					interDist = -(p1 - p2).norm();
				}
				//dir = -(p1 - p2) / interDist;

			}
		}
		OrientedBox3D OBB(center, u, v, w, extent);
		for (int i = 0; i < 18; i++)
		{
			Segment3D tmp;

			if (s_tet[i].intersect(OBB, tmp))
			{
				Point3D inp((tmp.startPoint() + tmp.endPoint()) / 2.0f);
				Point3D sp(tmp.startPoint());
				Point3D ep(tmp.endPoint());
				if (abs(inp.distance(OBB)) > abs(interDist))
				{
					p2 = inp.origin;
					p1 = inp.project(OBB).origin;
					interDist = -abs(inp.distance(OBB));
				}
				if (abs(sp.distance(OBB)) > abs(interDist))
				{
					p2 = sp.origin;
					p1 = sp.project(OBB).origin;
					interDist = -abs(sp.distance(OBB));
				}
				if (abs(ep.distance(OBB)) > abs(interDist))
				{
					p2 = ep.origin;
					p1 = ep.project(OBB).origin;
					interDist = -abs(ep.distance(OBB));
				}
			}
		}
		if (interDist > -EPSILON) return false;
		interNorm = (p1 - p2) / interDist;
		return true;
	}


	template<typename Real>
	DYN_FUNC bool TOrientedBox3D<Real>::point_intersect(const TTriangle3D<Real>& TRI, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2) const
	{

		Segment3D s_tri[3];
		s_tri[0] = Segment3D(TRI.v[0], TRI.v[1]);
		s_tri[1] = Segment3D(TRI.v[0], TRI.v[2]);
		s_tri[2] = Segment3D(TRI.v[2], TRI.v[1]);

		interDist = 0.0f;
		//obb intersect tet

		Point3D p[8];
		p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
		p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
		p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
		p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
		p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
		p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
		p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
		p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);

		Segment3D s[12];
		s[0] = Segment3D(p[0].origin, p[1].origin); s[1] = Segment3D(p[0].origin, p[2].origin); s[2] = Segment3D(p[0].origin, p[4].origin);
		s[3] = Segment3D(p[3].origin, p[1].origin); s[4] = Segment3D(p[3].origin, p[2].origin); s[5] = Segment3D(p[3].origin, p[7].origin);
		s[6] = Segment3D(p[6].origin, p[7].origin); s[7] = Segment3D(p[6].origin, p[2].origin); s[8] = Segment3D(p[6].origin, p[4].origin);
		s[9] = Segment3D(p[5].origin, p[7].origin); s[10] = Segment3D(p[5].origin, p[1].origin); s[11] = Segment3D(p[5].origin, p[4].origin);



		OrientedBox3D OBB(center, u, v, w, extent);
		for (int i = 0; i < 3; i++)
		{
			Segment3D tmp;

			if (s_tri[i].intersect(OBB, tmp))
			{
				Point3D inp((tmp.startPoint() + tmp.endPoint()) / 2.0f);
				Point3D sp(tmp.startPoint());
				Point3D ep(tmp.endPoint());
				if (abs(inp.distance(OBB)) > abs(interDist))
				{
					p2 = inp.origin;
					p1 = inp.project(OBB).origin;
					interDist = -abs(inp.distance(OBB));
				}
				if (abs(sp.distance(OBB)) > abs(interDist))
				{
					p2 = sp.origin;
					p1 = sp.project(OBB).origin;
					interDist = -abs(sp.distance(OBB));
				}
				if (abs(ep.distance(OBB)) > abs(interDist))
				{
					p2 = ep.origin;
					p1 = ep.project(OBB).origin;
					interDist = -abs(ep.distance(OBB));
				}
			}
		}
		interNorm = (p1 - p2) / interDist;
		if (interDist < -EPSILON)
			return true;

		for (int i = 0; i < 12; i++)
		{
			Point3D tmp_point;
			if (s[i].intersect(TRI, tmp_point))
			{
				Point3D tmp_p1;// = s[i].startPoint();
				Point3D tmp_p2;// = s[i].endPoint();
				if (tmp_point.distance(Point3D(s[i].startPoint())) < tmp_point.distance(Point3D(s[i].endPoint())))
				{
					tmp_p1 = s[i].endPoint();
					tmp_p2 = s[i].startPoint();
				}
				else
				{
					tmp_p1 = s[i].startPoint();
					tmp_p2 = s[i].endPoint();
				}
				if (abs(tmp_p2.distance(tmp_point)) > abs(interDist))
				{
					p2 = tmp_point.origin;
					p1 = tmp_p2.origin;
					interDist = -abs(tmp_p2.distance(tmp_point));
				}
			}
		}

		interNorm = (p1 - p2) / interDist;
		if (interDist < -EPSILON)
			return true;

		return false;
	}

}
