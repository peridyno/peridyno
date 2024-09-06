//#include "Primitive3D.h"
#include "Math/SimpleMath.h"
#include "Interval.h"
#include <glm/glm.hpp>

namespace dyno
{
	template<typename Real>
	DYN_FUNC TPoint2D<Real>::TPoint2D()
	{
		origin = Coord2D(0);
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real>::TPoint2D(const Real& val)
	{
		origin = Coord2D(val);
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real>::TPoint2D(const Real& c0, const Real& c1)
	{
		origin = Coord2D(c0, c1);
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real>::TPoint2D(const Coord2D& pos)
	{
		origin = pos;
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real>::TPoint2D(const TPoint2D& pt)
	{
		origin = pt.origin;
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real>& TPoint2D<Real>::operator=(const Coord2D& p)
	{
		origin = p;
		return *this;
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real> TPoint2D<Real>::project(const TLine2D<Real>& line) const
	{
		Coord2D u = origin - line.origin;
		Real tNum = u.dot(line.direction);
		Real a = line.direction.normSquared();
		Real t = a < REAL_EPSILON_SQUARED ? 0 : tNum / a;

		return TPoint2D(line.origin + t * line.direction);
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real> TPoint2D<Real>::project(const TRay2D<Real>& ray) const
	{
		Coord2D u = origin - ray.origin;

		Real tNum = u.dot(ray.direction);
		Real a = ray.direction.normSquared();
		Real t = a < REAL_EPSILON_SQUARED ? 0 : tNum / a;

		t = t < 0 ? 0 : t;

		return TPoint2D<Real>(ray.origin + t * ray.direction);
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real> TPoint2D<Real>::project(const TSegment2D<Real>& segment) const
	{
		Coord2D l = origin - segment.v0;
		Coord2D dir = segment.v1 - segment.v0;
		if (dir.normSquared() < REAL_EPSILON_SQUARED)
		{
			return TPoint2D<Real>(segment.v0);
		}

		Real t = l.dot(dir) / dir.normSquared();

		Coord2D q = segment.v0 + t * dir;
		q = t < 0 ? segment.v0 : q;
		q = t > 1 ? segment.v1 : q;
		//printf("T: %.3lf\n", t);
		return TPoint2D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC TPoint2D<Real> TPoint2D<Real>::project(const TCircle2D<Real>& circle) const
	{
		Coord2D cp = origin - circle.center;
		Coord2D q = circle.center + circle.radius * cp.normalize();

		return TPoint2D<Real>(q);
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distance(const TPoint2D<Real>& pt) const
	{
		return (origin - pt.origin).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distance(const TLine2D<Real>& line) const
	{
		return (origin - project(line).origin).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distance(const TRay2D<Real>& ray) const
	{
		return (origin - project(ray).origin).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distance(const TSegment2D<Real>& segment) const
	{
		return (origin - project(segment).origin).norm();
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distance(const TCircle2D<Real>& circle) const
	{
		return (origin - circle.center).norm() - circle.radius;
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distanceSquared(const TPoint2D& pt) const
	{
		return (origin - pt.origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distanceSquared(const TLine2D<Real>& line) const
	{
		return (origin - project(line).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distanceSquared(const TRay2D<Real>& ray) const
	{
		return (origin - project(ray).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distanceSquared(const TSegment2D<Real>& segment) const
	{
		return (origin - project(segment).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC Real TPoint2D<Real>::distanceSquared(const TCircle2D<Real>& circle) const
	{
		return (origin - project(circle).origin).normSquared();
	}

	template<typename Real>
	DYN_FUNC bool TPoint2D<Real>::inside(const TLine2D<Real>& line) const
	{
		if (!line.isValid())
		{
			return false;
		}

		return (origin - line.origin).cross(line.direction) < REAL_EPSILON_SQUARED;
	}

	template<typename Real>
	DYN_FUNC bool TPoint2D<Real>::inside(const TRay2D<Real>& ray) const
	{
		if (!inside(TLine2D<Real>(ray.origin, ray.direction)))
		{
			return false;
		}

		Coord2D offset = origin - ray.origin;
		Real t = offset.dot(ray.direction);

		return t > Real(0);
	}

	template<typename Real>
	DYN_FUNC bool TPoint2D<Real>::inside(const TSegment2D<Real>& segment) const
	{
		Coord2D dir = segment.direction();
		if (!inside(TLine2D<Real>(segment.startPoint(), dir)))
		{
			return false;
		}

		Coord2D offset = origin - segment.startPoint();
		Real t = offset.dot(dir) / dir.normSquared();

		return t > Real(0) && t < Real(1);
	}

	template<typename Real>
	DYN_FUNC bool TPoint2D<Real>::inside(const TCircle2D<Real>& circle) const
	{
		return (origin - circle.center).normSquared() < circle.radius * circle.radius;
	}

	template<typename Real>
	DYN_FUNC const TSegment2D<Real> TPoint2D<Real>::operator-(const TPoint2D<Real>& pt) const
	{
		return TSegment2D<Real>(pt.origin, origin);
	}

	template<typename Real>
	DYN_FUNC TLine2D<Real>::TLine2D()
	{
		origin = Coord2D(0);
		direction = Coord2D(1, 0, 0);
	}

	template<typename Real>
	DYN_FUNC TLine2D<Real>::TLine2D(const Coord2D& pos, const Coord2D& dir)
	{
		origin = pos;
		direction = dir;
	}

	template<typename Real>
	DYN_FUNC TLine2D<Real>::TLine2D(const TLine2D<Real>& line)
	{
		origin = line.origin;
		direction = line.direction;
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real> TLine2D<Real>::proximity(const TLine2D<Real>& line) const
	{
		Coord2D u = origin - line.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(line.direction);
		Real c = line.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(line.direction);
		Real f = u.normSquared();
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			TPoint2D<Real> p = TPoint2D<Real>(line.origin);
			return c < REAL_EPSILON ? p - p.project(*this) : TSegment2D<Real>(origin, line.origin + e / c * line.direction);
		}
		else
		{
			Real invDet = 1 / det;
			Real s = (b * e - c * d) * invDet;
			Real t = (a * e - b * d) * invDet;
			return TSegment2D<Real>(origin + s * direction, line.origin + t * line.direction);
		}
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real> TLine2D<Real>::proximity(const TRay2D<Real>& ray) const
	{
		Coord2D u = origin - ray.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(ray.direction);
		Real c = ray.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(ray.direction);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			TPoint2D<Real> p0(origin);
			TPoint2D<Real> p1(ray.origin);

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

		return TSegment2D<Real>(origin + (s * direction), ray.origin + (t * ray.direction));
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real> TLine2D<Real>::proximity(const TSegment2D<Real>& segment) const
	{
		Coord2D u = origin - segment.startPoint();
		Coord2D dir1 = segment.endPoint() - segment.startPoint();
		Real a = direction.normSquared();
		Real b = direction.dot(dir1);
		Real c = dir1.dot(dir1);
		Real d = u.dot(direction);
		Real e = u.dot(dir1);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			TPoint2D<Real> p0(origin);
			TPoint2D<Real> p1(segment.startPoint());

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

		return TSegment2D<Real>(origin + (s * direction), segment.startPoint() + (t * dir1));
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real> TLine2D<Real>::proximity(const TCircle2D<Real>& circle) const
	{
		Coord2D offset = circle.center - origin;
		Real d2 = direction.normSquared();
		if (d2 < REAL_EPSILON)
		{
			return TPoint2D<Real>(origin).project(circle) - TPoint2D<Real>(origin);
		}

		Coord2D p = origin + offset.dot(direction) / d2 * direction;

		return TPoint2D<Real>(p).project(circle) - TPoint2D<Real>(p);
	}

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::distance(const TPoint2D<Real>& pt) const
	{
		return pt.distance(*this);
	}

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::distance(const TLine2D<Real>& line) const
	{
		return proximity(line).length();
	}

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::distance(const TRay2D<Real>& ray) const
	{
		return proximity(ray).length();
	}

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::distance(const TSegment2D<Real>& segment) const
	{
		return proximity(segment).length();
	}

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::distanceSquared(const TPoint2D<Real>& pt) const
	{
		return pt.distanceSquared(*this);
	}

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::distanceSquared(const TLine2D<Real>& line) const
	{
		return proximity(line).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::distanceSquared(const TRay2D<Real>& ray) const
	{
		return proximity(ray).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::distanceSquared(const TSegment2D<Real>& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC int TLine2D<Real>::intersect(const TCircle2D<Real>& circle, TSegment2D<Real>& interSeg) const
	{
		Coord2D diff = origin - circle.center;
		Real a0 = diff.dot(diff) - circle.radius * circle.radius;
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

	template<typename Real>
	DYN_FUNC Real TLine2D<Real>::parameter(const Coord2D& pos) const
	{
		Coord2D l = pos - origin;
		Real d2 = direction.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(direction) / d2;
	}

	template<typename Real>
	DYN_FUNC bool TLine2D<Real>::isValid() const
	{
		return direction.normSquared() > REAL_EPSILON_SQUARED;
	}

	template<typename Real>
	DYN_FUNC TRay2D<Real>::TRay2D()
	{
		origin = Coord2D(0);
		direction = Coord2D(1, 0);
	}

	template<typename Real>
	DYN_FUNC TRay2D<Real>::TRay2D(const Coord2D& pos, const Coord2D& dir)
	{
		origin = pos;
		direction = dir;
	}

	template<typename Real>
	DYN_FUNC TRay2D<Real>::TRay2D(const TRay2D<Real>& ray)
	{
		origin = ray.origin;
		direction = ray.direction;
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real> TRay2D<Real>::proximity(const TRay2D<Real>& ray) const
	{
		Coord2D u = origin - ray.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(ray.direction);
		Real c = ray.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(ray.direction);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			TPoint2D<Real> p0(origin);
			TPoint2D<Real> p1(ray.origin);

			TPoint2D<Real> q0 = p0.project(ray);
			TPoint2D<Real> q1 = p1.project(*this);

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

		return TSegment2D<Real>(origin + (s * direction), ray.origin + (t * direction));
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real> TRay2D<Real>::proximity(const TSegment2D<Real>& segment) const
	{
		Coord2D u = origin - segment.startPoint();
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
				TPoint2D<Real> p0(origin);
				return p0.project(segment) - p0;
			}
			else
			{
				TPoint2D<Real> p1(segment.startPoint());
				TPoint2D<Real> p2(segment.endPoint());

				TPoint2D<Real> q1 = p1.project(*this);
				TPoint2D<Real> q2 = p2.project(*this);

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

		return TSegment2D<Real>(origin + (s * direction), segment.startPoint() + (t * segment.direction()));
	}

	template<typename Real>
	DYN_FUNC Real TRay2D<Real>::distance(const TPoint2D<Real>& pt) const
	{
		return pt.distance(*this);
	}

	template<typename Real>
	DYN_FUNC Real TRay2D<Real>::distance(const TSegment2D<Real>& segment) const
	{
		return proximity(segment).length();
	}

	template<typename Real>
	DYN_FUNC Real TRay2D<Real>::distanceSquared(const TPoint2D<Real>& pt) const
	{
		return pt.distanceSquared(*this);
	}

	template<typename Real>
	DYN_FUNC Real TRay2D<Real>::distanceSquared(const TSegment2D<Real>& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC int TRay2D<Real>::intersect(const TCircle2D<Real>& sphere, TSegment2D<Real>& interSeg) const
	{
		Coord2D diff = origin - sphere.center;
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
	DYN_FUNC Real TRay2D<Real>::parameter(const Coord2D& pos) const
	{
		Coord2D l = pos - origin;
		Real d2 = direction.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(direction) / d2;
	}

	template<typename Real>
	DYN_FUNC bool TRay2D<Real>::isValid() const
	{
		return direction.normSquared() > REAL_EPSILON_SQUARED;
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real>::TSegment2D()
	{
		v0 = Coord3D(0, 0);
		v1 = Coord3D(1, 0);
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real>::TSegment2D(const Coord2D& p0, const Coord2D& p1)
	{
		v0 = p0;
		v1 = p1;
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real>::TSegment2D(const TSegment2D<Real>& segment)
	{
		v0 = segment.v0;
		v1 = segment.v1;
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real> TSegment2D<Real>::proximity(const TSegment2D<Real>& segment) const
	{
		Coord2D u = v0 - segment.v0;
		Coord2D dir0 = v1 - v0;
		Coord2D dir1 = segment.v1 - segment.v0;
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
			TPoint2D<Real> p0 = l0 < l1 ? TPoint2D<Real>(v0) : TPoint2D<Real>(segment.v0);
			TPoint2D<Real> p1 = l0 < l1 ? TPoint2D<Real>(v1) : TPoint2D<Real>(segment.v1);
			TSegment2D<Real> longerSeg = l0 < l1 ? segment : *this;
			bool bOpposite = l0 < l1 ? false : true;
			TPoint2D<Real> q0 = p0.project(longerSeg);
			TPoint2D<Real> q1 = p1.project(longerSeg);
			TSegment2D<Real> ret = p0.distance(q0) < p1.distance(q1) ? (q0 - p0) : (q1 - p1);
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

		return TSegment2D<Real>(v0 + (s * dir0), segment.v0 + (t * dir1));
	}

	template<typename Real>
	DYN_FUNC Real TSegment2D<Real>::distance(const TSegment2D<Real>& segment) const
	{
		return proximity(segment).length();
	}

	template<typename Real>
	DYN_FUNC Real TSegment2D<Real>::distanceSquared(const TSegment2D<Real>& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	template<typename Real>
	DYN_FUNC Real TSegment2D<Real>::length() const
	{
		return (v1 - v0).norm();
	}

	template<typename Real>
	DYN_FUNC Real TSegment2D<Real>::lengthSquared() const
	{
		return (v1 - v0).normSquared();
	}

	template<typename Real>
	DYN_FUNC int TSegment2D<Real>::intersect(const TCircle2D<Real>& circle, TSegment2D<Real>& interSeg) const
	{
		Coord2D diff = v0 - circle.center;
		Coord2D dir = direction();
		Real a0 = diff.dot(diff) - circle.radius * circle.radius;
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
	DYN_FUNC Real TSegment2D<Real>::parameter(const Coord2D& pos) const
	{
		Coord2D l = pos - v0;
		Coord2D dir = direction();
		Real d2 = dir.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(dir) / d2;
	}

	template<typename Real>
	DYN_FUNC bool TSegment2D<Real>::isValid() const
	{
		return lengthSquared() >= REAL_EPSILON_SQUARED;
	}

	template<typename Real>
	DYN_FUNC TSegment2D<Real> TSegment2D<Real>::operator-(void) const
	{
		TSegment2D<Real> seg;
		seg.v0 = v1;
		seg.v1 = v0;

		return seg;
	}

	template<typename Real>
	DYN_FUNC TCircle2D<Real>::TCircle2D()
	{
		center = Coord2D(0);
		radius = Real(1);
		theta = Real(0);
	}

	template<typename Real>
	DYN_FUNC TCircle2D<Real>::TCircle2D(const Coord2D& c, const Real& r)
	{
		center = c;
		radius = r;
		theta = Real(0);
	}

	template<typename Real>
	DYN_FUNC TCircle2D<Real>::TCircle2D(const TCircle2D<Real>& circle)
	{
		center = circle.center;
		radius = circle.radius;
		theta = circle.theta;
	}

	template<typename Real>
	DYN_FUNC TAlignedBox2D<Real>::TAlignedBox2D()
	{
		v0 = Coord2D(0);
		v1 = Coord2D(1);
	}

	template<typename Real>
	DYN_FUNC TAlignedBox2D<Real>::TAlignedBox2D(const Coord2D& p0, const Coord2D& p1)
	{
		v0 = p0;
		v1 = p1;
	}

	template<typename Real>
	DYN_FUNC TAlignedBox2D<Real>::TAlignedBox2D(const TAlignedBox2D<Real>& box)
	{
		v0 = box.v0;
		v1 = box.v1;
	}

	template<typename Real>
	TPolygon2D<Real>::TPolygon2D()
	{
		_center = Coord2D(0);
	}

	template<typename Real>
	TPolygon2D<Real>::~TPolygon2D()
	{

	}


	template<typename Real>
	void TPolygon2D<Real>::setAsBox(Real hx, Real hy)
	{
		size = 4;

		_vertices[0] = Coord2D(-hx, -hy);
		_vertices[1] = Coord2D(hx, -hy);
		_vertices[2] = Coord2D(hx, hy);
		_vertices[3] = Coord2D(-hx, hy);

		_normals[0] = Coord2D(Real(0), -Real(1));
		_normals[1] = Coord2D(Real(1), Real(0));
		_normals[2] = Coord2D(Real(0), Real(1));
		_normals[3] = Coord2D(-Real(1), Real(0));

		_center = Coord2D(0);
	}

	template<typename Real>
	void TPolygon2D<Real>::setAsPentagon(const Coord2D& v0, const Coord2D& v1, const Coord2D& v2, const Coord2D& v3, const Coord2D& v4)
	{
		size = 5;

		_vertices[0] = v0;
		_vertices[1] = v1;
		_vertices[2] = v2;
		_vertices[3] = v3;
		_vertices[4] = v4;

		_center = Coord2D(0);
	}

	template<typename Real>
	void TPolygon2D<Real>::setAsTriangle(const Coord2D& v0, const Coord2D& v1, const Coord2D& v2)
	{
		size = 3;

		_vertices[0] = v0;
		_vertices[1] = v1;
		_vertices[2] = v2;

		_center = Coord2D(0);
	}

	template<typename Real>
	void dyno::TPolygon2D<Real>::setAsLine(const Coord2D& v0, const Coord2D& v1)
	{
		size = 2;

		_vertices[0] = v0;
		_vertices[1] = v1;

		_center = Coord2D(0);
	}

	template<typename Real>
	TAlignedBox2D<Real> TPolygon2D<Real>::aabb()
	{
		Coord2D v0(REAL_MAX);
		Coord2D v1(-REAL_MAX);
		for (uint i = 0; i < size; i++)
		{
			Coord2D v = _vertices[i] + _center;
			v0 = v0.minimum(v);
			v1 = v1.maximum(v);
		}

		return TAlignedBox2D<Real>(v0 - _radius, v1 + _radius);
	}
}
