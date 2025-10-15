#pragma once
/**
 * @file Primitive3D.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-02-24
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"

// Temporary code, to be removed in the  
namespace px
{
	struct Plane3D
	{
		dyno::Vec3f origin;
		dyno::Vec3f normal;
	};

	struct Sphere3D
	{
		float radius;
		dyno::Vec3f center;
	};

	struct Box
	{
		Box()
		{
			center = dyno::Vec3f(0.0f, 0.0f, 0.0f);
			halfLength = dyno::Vec3f(1.0f, 1.0f, 1.0f);
			rot = dyno::Quat1f(1.0f, 0.0f, 0.0f, 0.0f);
		}

		dyno::Vec3f center;
		dyno::Vec3f halfLength;

		dyno::Quat1f rot;
	};

	struct alignas(16) Sphere
	{
		dyno::Quat1f rot;
		glm::vec3 center;
		float radius;
	};

	struct alignas(16) Capsule
	{
		dyno::Quat1f rot;
		dyno::Vec3f center;
		float halfLength;
		float radius;
	};

	struct Ray3D
	{
		dyno::Vec3f origin;
		dyno::Vec3f direction;
	};
}

namespace dyno
{
	/**
	 * @brief 0D geometric primitive in three-dimensional space
	 *
	 */
	template <typename Real> class TPoint3D;

	/**
	 * @brief 1D geometric primitives in three-dimensional space
	 *
	 */
	template <typename Real> class TLine3D;
	template <typename Real> class TRay3D;
	template <typename Real> class TSegment3D;

	/**
	 * @brief 2D geometric primitives in three-dimensional space
	 *
	 */
	template <typename Real> class TPlane3D;
	template <typename Real> class TTriangle3D;
	template <typename Real> class TRectangle3D;
	template <typename Real> class TDisk3D;

	/**
	 * @brief 3D geometric primitives in three-dimensional space
	 *
	 */
	template <typename Real> class TSphere3D;
	template <typename Real> class TCapsule3D;
	template <typename Real> class TTet3D;
	template <typename Real> class TAlignedBox3D;
	template <typename Real> class TOrientedBox3D;
	template <typename Real> class TCylinder3D;
	template <typename Real> class TCone3D;
	template <typename Real> class TGrid3D;
	template <typename Real> class TMedialCone3D;
	template <typename Real> class TMedialSlab3D;

	template<typename Real>
	class TPoint3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TPoint3D();
		DYN_FUNC TPoint3D(const Real& c0, const Real& c1, const Real& c2);
		DYN_FUNC TPoint3D(const TPoint3D& pt);

		DYN_FUNC TPoint3D& operator = (const Coord3D& p);

		explicit DYN_FUNC TPoint3D(const Real& val);
		explicit DYN_FUNC TPoint3D(const Coord3D& pos);


		/**
		 * @brief project a point onto linear components -- lines, rays and segments
		 *
		 * @param line/ray/segment linear components
		 * @return closest point
		 */
		DYN_FUNC TPoint3D<Real> project(const TLine3D<Real>& line) const;
		DYN_FUNC TPoint3D<Real> project(const TRay3D<Real>& ray) const;
		DYN_FUNC TPoint3D<Real> project(const TSegment3D<Real>& segment) const;
		/**
		 * @brief project a point onto planar components -- planes, triangles and disks
		 *
		 * @param plane/triangle/disk planar components
		 * @return closest point
		 */
		DYN_FUNC TPoint3D<Real> project(const TPlane3D<Real>& plane) const;
		DYN_FUNC TPoint3D<Real> project(const TTriangle3D<Real>& triangle) const;
		DYN_FUNC TPoint3D<Real> project(const TRectangle3D<Real>& rectangle) const;
		DYN_FUNC TPoint3D<Real> project(const TDisk3D<Real>& disk) const;
		/**
		 * @brief project a point onto polyhedra components -- tetrahedra, spheres and oriented bounding boxes
		 *
		 * @param sphere/capsule/tet/abox/obb polyhedra components
		 * @return closest point
		 */
		DYN_FUNC TPoint3D<Real> project(const TSphere3D<Real>& sphere) const;
		DYN_FUNC TPoint3D<Real> project(const TCapsule3D<Real>& capsule) const;
		DYN_FUNC TPoint3D<Real> project(const TTet3D<Real>& tet) const;
		DYN_FUNC TPoint3D<Real> project(const TAlignedBox3D<Real>& abox) const;
		DYN_FUNC TPoint3D<Real> project(const TOrientedBox3D<Real>& obb) const;

		DYN_FUNC TPoint3D<Real> project(const TSphere3D<Real>& sphere, Bool& bInside) const;
		DYN_FUNC TPoint3D<Real> project(const TCapsule3D<Real>& capsule, Bool& bInside) const;
		DYN_FUNC TPoint3D<Real> project(const TTet3D<Real>& tet, Bool& bInside) const;
		DYN_FUNC TPoint3D<Real> project(const TTet3D<Real>& tet, Bool& bInside, int* idx) const;
		DYN_FUNC TPoint3D<Real> project(const TAlignedBox3D<Real>& abox, Bool& bInside) const;
		DYN_FUNC TPoint3D<Real> project(const TOrientedBox3D<Real>& obb, Bool& bInside) const;



		DYN_FUNC Real distance(const TPoint3D<Real>& pt) const;
		DYN_FUNC Real distance(const TLine3D<Real>& line) const;
		DYN_FUNC Real distance(const TRay3D<Real>& ray) const;
		DYN_FUNC Real distance(const TSegment3D<Real>& segment) const;
		/**
		 * @brief compute the signed distance to 2D geometric primitives
		 *
		 * @param plane/triangle/rectangle/disk planar components
		 * @return positive if point resides in the positive side of 2D geometric primitives
		 */
		DYN_FUNC Real distance(const TPlane3D<Real>& plane) const;
		DYN_FUNC Real distance(const TTriangle3D<Real>& triangle) const;
		DYN_FUNC Real distance(const TRectangle3D<Real>& rectangle) const;
		DYN_FUNC Real distance(const TDisk3D<Real>& disk) const;

		/**
		 * @brief compute signed distance to 3D geometric primitives
		 *
		 * @param sphere/tet/abox/obb 3D geometric primitives
		 * @return Real negative distance if a point is inside the 3D geometric primitive, otherwise return a positive value
		 */
		DYN_FUNC Real distance(const TSphere3D<Real>& sphere) const;
		DYN_FUNC Real distance(const TCapsule3D<Real>& capsule) const;
		DYN_FUNC Real distance(const TTet3D<Real>& tet) const;
		DYN_FUNC Real distance(const TAlignedBox3D<Real>& abox) const;
		DYN_FUNC Real distance(const TOrientedBox3D<Real>& obb) const;



		DYN_FUNC Real distanceSquared(const TPoint3D<Real>& pt) const;
		DYN_FUNC Real distanceSquared(const TLine3D<Real>& line) const;
		DYN_FUNC Real distanceSquared(const TRay3D<Real>& ray) const;
		DYN_FUNC Real distanceSquared(const TSegment3D<Real>& segment) const;
		/**
		 * @brief return squared distance from a point to 3D geometric primitives
		 *
		 * @param plane/triangle/rectangle/disk planar components
		 * @return DYN_FUNC distanceSquared
		 */
		DYN_FUNC Real distanceSquared(const TPlane3D<Real>& plane) const;
		DYN_FUNC Real distanceSquared(const TTriangle3D<Real>& triangle) const;
		DYN_FUNC Real distanceSquared(const TRectangle3D<Real>& rectangle) const;
		DYN_FUNC Real distanceSquared(const TDisk3D<Real>& disk) const;
		/**
		 * @brief return squared distance from a point to 3D geometric primitives
		 *
		 * @param sphere/capsule/tet/abox/obb 3D geometric primitives
		 * @return Real squared distance
		 */
		DYN_FUNC Real distanceSquared(const TSphere3D<Real>& sphere) const;
		DYN_FUNC Real distanceSquared(const TCapsule3D<Real>& capsule) const;
		DYN_FUNC Real distanceSquared(const TTet3D<Real>& tet) const;
		DYN_FUNC Real distanceSquared(const TAlignedBox3D<Real>& abox) const;
		DYN_FUNC Real distanceSquared(const TOrientedBox3D<Real>& obb) const;


		/**
		 * @brief check whether a point strictly lies inside (excluding boundary) a 1D geometric primitive
		 *
		 * @param line/ray/segment 1D geometric primitives
		 * @return true if a point is inside the geometric primitive, otherwise return false
		 */
		DYN_FUNC bool inside(const TLine3D<Real>& line) const;
		DYN_FUNC bool inside(const TRay3D<Real>& ray) const;
		DYN_FUNC bool inside(const TSegment3D<Real>& segment) const;
		/**
		 * @brief check whether a point strictly lies inside (excluding boundary) a 2D geometric primitive
		 *
		 * @param plane/triangle/rectangle/disk 2D geometric primitives
		 * @return true if a point is inside the geometric primitive, otherwise return false
		 */
		DYN_FUNC bool inside(const TPlane3D<Real>& plane) const;
		DYN_FUNC bool inside(const TTriangle3D<Real>& triangle) const;
		DYN_FUNC bool inside(const TRectangle3D<Real>& rectangle) const;
		DYN_FUNC bool inside(const TDisk3D<Real>& disk) const;
		/**
		 * @brief check whether a point strictly lies inside (excluding boundary) a 3D geometric primitive
		 *
		 * @param sphere/tet/abox/obb 3D geometric primitives
		 * @return true if a point is inside the geometric primitive, otherwise return false
		 */
		DYN_FUNC bool inside(const TSphere3D<Real>& sphere) const;
		DYN_FUNC bool inside(const TCapsule3D<Real>& capsule) const;
		DYN_FUNC bool inside(const TTet3D<Real>& tet) const;
		DYN_FUNC bool inside(const TAlignedBox3D<Real>& box) const;
		DYN_FUNC bool inside(const TOrientedBox3D<Real>& obb) const;

		DYN_FUNC const TSegment3D<Real> operator-(const TPoint3D<Real>& pt) const;

		Coord3D origin;
	};

	template<typename Real>
	class TLine3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TLine3D();
		/**
		 * @brief
		 *
		 * @param pos
		 * @param dir = 0 indicate the line degenerates into a point
		 */
		DYN_FUNC TLine3D(const Coord3D& pos, const Coord3D& dir);
		DYN_FUNC TLine3D(const TLine3D<Real>& line);

		DYN_FUNC TSegment3D<Real> proximity(const TLine3D<Real>& line) const;
		DYN_FUNC TSegment3D<Real> proximity(const TRay3D<Real>& ray) const;
		DYN_FUNC TSegment3D<Real> proximity(const TSegment3D<Real>& segment) const;

		DYN_FUNC TSegment3D<Real> proximity(const TTriangle3D<Real>& triangle) const;
		DYN_FUNC TSegment3D<Real> proximity(const TRectangle3D<Real>& rectangle) const;

		DYN_FUNC TSegment3D<Real> proximity(const TSphere3D<Real>& sphere) const;
		DYN_FUNC TSegment3D<Real> proximity(const TAlignedBox3D<Real>& box) const;
		DYN_FUNC TSegment3D<Real> proximity(const TOrientedBox3D<Real>& obb) const;


		DYN_FUNC Real distance(const TPoint3D<Real>& pt) const;
		DYN_FUNC Real distance(const TLine3D<Real>& line) const;
		DYN_FUNC Real distance(const TRay3D<Real>& ray) const;
		DYN_FUNC Real distance(const TSegment3D<Real>& segment) const;

		DYN_FUNC Real distance(const TAlignedBox3D<Real>& box) const;
		DYN_FUNC Real distance(const TOrientedBox3D<Real>& obb) const;

		// 		DYN_FUNC Line3D(const Coord3D& pos, const Coord3D& dir);
		// 		DYN_FUNC Line3D(const Line3D& line);


		DYN_FUNC Real distanceSquared(const TPoint3D<Real>& pt) const;
		DYN_FUNC Real distanceSquared(const TLine3D<Real>& line) const;
		DYN_FUNC Real distanceSquared(const TRay3D<Real>& ray) const;
		DYN_FUNC Real distanceSquared(const TSegment3D<Real>& segment) const;
		/**
		 * @brief compute signed distance to 3D geometric primitives
		 *
		 * @param box/obb
		 * @return 0 if intersecting the 3D geometric primitives
		 */
		DYN_FUNC Real distanceSquared(const TAlignedBox3D<Real>& box) const;
		DYN_FUNC Real distanceSquared(const TOrientedBox3D<Real>& obb) const;

		/**
		 * @brief intersection tests
		 * 
		 * @return 0 if there is no intersection
		 */
		DYN_FUNC int intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const;
		DYN_FUNC int intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const;

		DYN_FUNC int intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const;
		DYN_FUNC int intersect(const TTet3D<Real>& tet, TSegment3D<Real>& interSeg) const;
		DYN_FUNC int intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const;
		DYN_FUNC int intersect(const TOrientedBox3D<Real>& obb, TSegment3D<Real>& interSeg) const;


		DYN_FUNC Real parameter(const Coord3D& pos) const;

		DYN_FUNC bool isValid() const;

		Coord3D origin;

		//direction will be normalized during construction
		Coord3D direction;
	};

	template<typename Real>
	class TRay3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TRay3D();

		struct Param
		{
			Real t;
		};

		/**
		 * @brief
		 *
		 * @param pos
		 * @param ||dir|| = 0 indicates the ray degenerates into a point
		 * @return DYN_FUNC
		 */
		DYN_FUNC TRay3D(const Coord3D& pos, const Coord3D& dir);
		DYN_FUNC TRay3D(const TRay3D<Real>& ray);

		DYN_FUNC TSegment3D<Real> proximity(const TRay3D<Real>& ray) const;
		DYN_FUNC TSegment3D<Real> proximity(const TSegment3D<Real>& segment) const;

		DYN_FUNC TSegment3D<Real> proximity(const TTriangle3D<Real>& triangle) const;
		DYN_FUNC TSegment3D<Real> proximity(const TRectangle3D<Real>& rectangle) const;

		DYN_FUNC TSegment3D<Real> proximity(const TAlignedBox3D<Real>& box) const;
		DYN_FUNC TSegment3D<Real> proximity(const TOrientedBox3D<Real>& obb) const;

		DYN_FUNC Real distance(const TPoint3D<Real>& pt) const;
		DYN_FUNC Real distance(const TSegment3D<Real>& segment) const;
		DYN_FUNC Real distance(const TTriangle3D<Real>& triangle) const;

		DYN_FUNC Real distanceSquared(const TPoint3D<Real>& pt) const;
		DYN_FUNC Real distanceSquared(const TSegment3D<Real>& segment) const;
		DYN_FUNC Real distanceSquared(const TTriangle3D<Real>& triangle) const;

		DYN_FUNC int intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const;
		DYN_FUNC int intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const;

		DYN_FUNC int intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const;

		DYN_FUNC int intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const;
		DYN_FUNC int intersect(const TOrientedBox3D<Real>& obb, TSegment3D<Real>& interSeg) const;

		DYN_FUNC Real parameter(const Coord3D& pos) const;

		DYN_FUNC bool isValid() const;

		Coord3D origin;

		//guarantee direction is a unit vector
		Coord3D direction;
	};

	template<typename Real>
	class TSegment3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TSegment3D();
		DYN_FUNC TSegment3D(const Coord3D& p0, const Coord3D& p1);
		DYN_FUNC TSegment3D(const TSegment3D<Real>& segment);

		/**
		 * @brief return a segment pointing from the input segment to the other primitive
		 */
		DYN_FUNC TSegment3D<Real> proximity(const TSegment3D<Real>& segment) const;

		DYN_FUNC TSegment3D<Real> proximity(const TPlane3D<Real>& plane) const;
		DYN_FUNC TSegment3D<Real> proximity(const TTriangle3D<Real>& triangle) const;
		DYN_FUNC TSegment3D<Real> proximity(const TRectangle3D<Real>& rectangle) const;

		DYN_FUNC TSegment3D<Real> proximity(const TAlignedBox3D<Real>& box) const;
		DYN_FUNC TSegment3D<Real> proximity(const TOrientedBox3D<Real>& obb) const;
		DYN_FUNC TSegment3D<Real> proximity(const TTet3D<Real>& tet) const;

		DYN_FUNC Real distance(const TSegment3D<Real>& segment) const;

		DYN_FUNC Real distance(const TTriangle3D<Real>& triangle) const;

		DYN_FUNC Real distanceSquared(const TSegment3D<Real>& segment) const;

		DYN_FUNC Real distanceSquared(const TTriangle3D<Real>& triangle) const;

		DYN_FUNC bool intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const;
		DYN_FUNC bool intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const;

		DYN_FUNC int intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const;
		DYN_FUNC int intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const;
		DYN_FUNC int intersect(const TOrientedBox3D<Real>& obb, TSegment3D<Real>& interSeg) const;
		
		DYN_FUNC Real length() const;
		DYN_FUNC Real lengthSquared() const;

		DYN_FUNC Real parameter(const Coord3D& pos) const;

		inline DYN_FUNC Coord3D& startPoint() { return v0; }
		inline DYN_FUNC Coord3D& endPoint() { return v1; }

		inline DYN_FUNC Coord3D startPoint() const { return v0; }
		inline DYN_FUNC Coord3D endPoint() const { return v1; }

		inline DYN_FUNC Coord3D direction() const { return v1 - v0;	}

		inline DYN_FUNC TSegment3D<Real> operator-(void) const;

		DYN_FUNC bool isValid() const;

		Coord3D v0;
		Coord3D v1;
	};

	template<typename Real>
	class TPlane3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TPlane3D();
		DYN_FUNC TPlane3D(const Coord3D& pos, const Coord3D n);
		DYN_FUNC TPlane3D(const TPlane3D& plane);

		DYN_FUNC bool isValid() const;

		Coord3D origin;

		/**
		 * @brief the plane will be treated as a single point if its normal is zero
		 */
		Coord3D normal;
	};


	template<typename Real>
	class TTriangle3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TTriangle3D();
		DYN_FUNC TTriangle3D(const Coord3D& p0, const Coord3D& p1, const Coord3D& p2);
		DYN_FUNC TTriangle3D(const TTriangle3D& triangle);

		struct Param
		{
			Real u;
			Real v;
			Real w;
		};

		DYN_FUNC Real area() const;
		DYN_FUNC Coord3D normal() const;

		DYN_FUNC bool computeBarycentrics(const Coord3D& p, Param& bary) const;
		DYN_FUNC Coord3D computeLocation(const Param& bary) const;

		DYN_FUNC Real maximumEdgeLength() const;

		DYN_FUNC bool isValid() const;

		DYN_FUNC TAlignedBox3D<Real> aabb();

		DYN_FUNC Real distanceSquared(const TTriangle3D<Real>& triangle) const;

		DYN_FUNC Real distance(const TTriangle3D<Real>& triangle) const;

		Coord3D v[3];
	};

	template<typename Real>
	class TRectangle3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TRectangle3D();
		DYN_FUNC TRectangle3D(const Coord3D& c, const Coord3D& a0, const Coord3D& a1, const Coord2D& ext);
		DYN_FUNC TRectangle3D(const TRectangle3D<Real>& rectangle);

		struct Param
		{
			Real u;
			Real v;
		};

		DYN_FUNC TPoint3D<Real> vertex(const int i) const;
		DYN_FUNC TSegment3D<Real> edge(const int i) const;

		DYN_FUNC Real area() const;
		DYN_FUNC Coord3D normal() const;

		DYN_FUNC bool computeParams(const Coord3D& p, Param& par) const;

		DYN_FUNC bool isValid() const;

		Coord3D center;
		/**
		 * @brief two orthonormal unit axis
		 *
		 */
		Coord3D axis[2];
		Coord2D extent;
	};

	template<typename Real>
	class TDisk3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TDisk3D();
		DYN_FUNC TDisk3D(const Coord3D& c, const Coord3D& n, const Real& r);
		DYN_FUNC TDisk3D(const TDisk3D<Real>& circle);

		DYN_FUNC Real area();

		DYN_FUNC bool isValid();

		Real radius;
		Coord3D center;
		Coord3D normal;
	};

	template<typename Real>
	class TSphere3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TSphere3D();
		DYN_FUNC TSphere3D(const Coord3D& c, const Real& r);
		DYN_FUNC TSphere3D(const Coord3D& c, const Quat<Real>& rot, const Real& r);
		DYN_FUNC TSphere3D(const TSphere3D<Real>& sphere);

		DYN_FUNC Real volume();

		DYN_FUNC bool isValid();

		DYN_FUNC TAlignedBox3D<Real> aabb();

		Coord3D center;
		Quat<Real> rotation;
		Real radius;
	};

	template<typename Real>
	class TCylinder3D
	{
	public:
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TCylinder3D();
		DYN_FUNC TCylinder3D(const Coord3D& c, const Real& h, const Real &r, const Quat<Real>& rot = Quat<Real>(), const Coord3D& s = Coord3D(1));
		DYN_FUNC TCylinder3D(const TCylinder3D<Real>& cylinder);
		DYN_FUNC Real volume() const { return Real(M_PI) * radius * radius * height * scale[0] * scale[1] * scale[2]; }

		Coord3D center;
		Real height;
		Real radius;

		Coord3D scale;
		Quat<Real> rotation;
	};

	template<typename Real>
	class TCone3D
	{
	public:
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TCone3D();
		DYN_FUNC TCone3D(const Coord3D& c, const Real& h, const Real& r, const Quat<Real>& rot = Quat<Real>(), const Coord3D& s = Coord3D(1));
		DYN_FUNC TCone3D(const TCone3D<Real>& cone);
		DYN_FUNC Real volume() const { return Real(M_PI) * radius * radius * height / Real(3); }

		//Center of the bottom circle
		Coord3D center;
		Real height;
		Real radius;

		Coord3D scale;
		Quat<Real> rotation;
	};

	// The centerline is set to align with the Y-axis in default 
	template<typename Real>
	class TCapsule3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TCapsule3D();
		DYN_FUNC TCapsule3D(const Coord3D& c, const Quat<Real>& q, const Real& r, const Real& hl);
		DYN_FUNC TCapsule3D(const Coord3D& v0, const Coord3D& v1, const Real& r);
		DYN_FUNC TCapsule3D(const TCapsule3D<Real>& capsule);

		DYN_FUNC Real volume() const;

		DYN_FUNC bool isValid() const;
		DYN_FUNC TAlignedBox3D<Real> aabb() const;

		// return the two ends
		DYN_FUNC inline Coord3D startPoint() const { return center - rotation.rotate(halfLength * Coord3D(0, 1, 0)); }
		DYN_FUNC inline Coord3D endPoint() const { return center + rotation.rotate(halfLength * Coord3D(0, 1, 0)); }

		DYN_FUNC inline TSegment3D<Real> centerline() const { return TSegment3D<Real>(startPoint(), endPoint()); }

		Coord3D center;
		Quat<Real> rotation;
		Real radius;
		Real halfLength;
	};

	/**
	 * @brief vertices are ordered so that the normal vectors for the triangular faces point outwards
	 *			3
	 *        /  | \
	 *       0 - 2 - 1
	 */
	template<typename Real>
	class TTet3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;
		typedef  SquareMatrix<Real, 3> Matrix3D;

	public:
		DYN_FUNC TTet3D();
		DYN_FUNC TTet3D(const Coord3D& v0, const Coord3D& v1, const Coord3D& v2, const Coord3D& v3);
		DYN_FUNC TTet3D(const TTet3D<Real>& tet);

		DYN_FUNC TTriangle3D<Real> face(const int index) const;
		DYN_FUNC TSegment3D<Real> edge(const int index) const;

		DYN_FUNC Real solidAngle(const int index) const;

		DYN_FUNC Real volume() const;

		DYN_FUNC bool isValid() const;

		//DYN_FUNC bool intersect(const TTet3D<Real>& tet, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance = 1) const;
		DYN_FUNC bool intersect(const TTet3D<Real>& tet, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance = 1) const;
		DYN_FUNC bool intersect(const TTriangle3D<Real>& tri, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance = 1) const;

		DYN_FUNC void expand(Real r);
		DYN_FUNC TAlignedBox3D<Real> aabb();

		// http://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter
		DYN_FUNC TPoint3D<Real> circumcenter() const;
		DYN_FUNC TPoint3D<Real> barycenter() const;

		DYN_FUNC bool contain(Coord3D p);

		DYN_FUNC Vector<Real, 4> computeBarycentricCoordinates(const Coord3D& p);

		Coord3D v[4];
	};


	template<typename Real>
	class TMedialCone3D{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;
		typedef  SquareMatrix<Real, 3> Matrix3D;

	public:
		DYN_FUNC TMedialCone3D();
		DYN_FUNC TMedialCone3D(const Coord3D& v0, const Coord3D& v1, const Real& r0, const Real& r1);
		DYN_FUNC TMedialCone3D(const TMedialCone3D<Real>& cone);

		DYN_FUNC Real volume() const;
		DYN_FUNC bool isValid() const;

		DYN_FUNC TAlignedBox3D<Real> aabb() const;

		Coord3D v[2];
		Real radius[2];
	};

	template<typename Real>
	class TMedialSlab3D{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;
		typedef  SquareMatrix<Real, 3> Matrix3D;

	public:
		DYN_FUNC TMedialSlab3D();
		DYN_FUNC TMedialSlab3D(const Coord3D& v0, const Coord3D& v1, const Coord3D& v2, const Real& r0, const Real& r1, const Real& r2);
		DYN_FUNC TMedialSlab3D(const TMedialSlab3D<Real>& slab);

		DYN_FUNC Real volume() const;
		DYN_FUNC bool isValid() const;

		DYN_FUNC TAlignedBox3D<Real> aabb() const;

		Coord3D v[3];
		Real radius[3];
	};

	template<typename Real>
	class TAlignedBox3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;
		typedef  SquareMatrix<Real, 3> Matrix3D;

	public:
		DYN_FUNC TAlignedBox3D();
		DYN_FUNC TAlignedBox3D(const Coord3D& p0, const Coord3D& p1);
		DYN_FUNC TAlignedBox3D(const TAlignedBox3D<Real>& box);

		DYN_FUNC bool intersect(const TAlignedBox3D<Real>& abox, TAlignedBox3D<Real>& interBox) const;
		DYN_FUNC bool checkOverlap(const TAlignedBox3D<Real>& abox) const;

		DYN_FUNC TAlignedBox3D<Real> merge(const TAlignedBox3D<Real>& aabb) const;

		DYN_FUNC bool meshInsert(const TTriangle3D<Real>& tri) const;
		DYN_FUNC bool isValid();

		DYN_FUNC TOrientedBox3D<Real> rotate(const Matrix3D& mat);

		DYN_FUNC inline Real length(unsigned int i) const { return v1[i] - v0[i]; }

		Coord3D v0;
		Coord3D v1;
	};

	template<typename Real>
	class TOrientedBox3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;
		typedef  SquareMatrix<Real, 3> Matrix3D;

	public:
		DYN_FUNC TOrientedBox3D();

		/**
		 * @brief construct an oriented bounding box, gurantee u_t, v_t and w_t are unit vectors and form right-handed orthornormal basis
		 *
		 * @param c  centerpoint
		 * @param u_t
		 * @param v_t
		 * @param w_t
		 * @param ext half the dimension in each of the u, v, and w directions
		 * @return DYN_FUNC
		 */
		DYN_FUNC TOrientedBox3D(const Coord3D c, const Coord3D u_t, const Coord3D v_t, const Coord3D w_t, const Coord3D ext);

		DYN_FUNC TOrientedBox3D(const Coord3D c, const Quat<Real> rot, const Coord3D ext);

		DYN_FUNC TOrientedBox3D(const TOrientedBox3D<Real>& obb);

		DYN_FUNC TPoint3D<Real> vertex(const int i) const;
		DYN_FUNC TSegment3D<Real> edge(const int i) const;
		DYN_FUNC TRectangle3D<Real> face(const int i) const;

		DYN_FUNC Real volume();

		DYN_FUNC bool isValid();

		DYN_FUNC TOrientedBox3D<Real> rotate(const Matrix3D& mat);

		DYN_FUNC TAlignedBox3D<Real> aabb();

		//DYN_FUNC bool point_intersect(const TOrientedBox3D<Real>& OBB, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2) const;
		DYN_FUNC bool point_intersect(const TOrientedBox3D<Real>& OBB, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2) const;
		DYN_FUNC bool point_intersect(const TTet3D<Real>& TET, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2) const;
		DYN_FUNC bool point_intersect(const TTriangle3D<Real>& TRI, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2) const;


		//DYN_FUNC bool triangle_intersect(const TTriangle3D<Real>& Tri) const;
		/**
		 * @brief centerpoint
		 *
		 */
		Coord3D center;

		/**
		 * @brief three unit vectors u, v and w forming a right-handed orthornormal basis
		 *
		 */
		Coord3D u, v, w;

		/**
		 * @brief half the dimension in each of the u, v, and w directions
		 */
		Coord3D extent;
	};

	template class TPoint3D<float>;
	template class TLine3D<float>;
	template class TRay3D<float>;
	template class TSegment3D<float>;
	template class TPlane3D<float>;
	template class TTriangle3D<float>;
	template class TRectangle3D<float>;
	template class TDisk3D<float>;
	template class TSphere3D<float>;
	template class TCapsule3D<float>;
	template class TTet3D<float>;
	template class TAlignedBox3D<float>;
	template class TOrientedBox3D<float>;
	template class TCylinder3D<float>;
	template class TCone3D<float>;
	template class TMedialCone3D<float>;
	template class TMedialSlab3D<float>;

	template class TPoint3D<double>;
	template class TLine3D<double>;
	template class TRay3D<double>;
	template class TSegment3D<double>;
	template class TPlane3D<double>;
	template class TTriangle3D<double>;
	template class TRectangle3D<double>;
	template class TDisk3D<double>;
	template class TSphere3D<double>;
	template class TCapsule3D<double>;
	template class TTet3D<double>;
	template class TAlignedBox3D<double>;
	template class TOrientedBox3D<double>;
	template class TCylinder3D<double>;
	template class TCone3D<double>;

#ifdef PRECISION_FLOAT
	//convenient typedefs 
	typedef TPoint3D<float> Point3D;
	typedef TLine3D<float> Line3D;
	typedef TRay3D<float> Ray3D;
	typedef TSegment3D<float> Segment3D;
	typedef TPlane3D<float> Plane3D;
	typedef TTriangle3D<float> Triangle3D;
	typedef TRectangle3D<float> Rectangle3D;
	typedef TDisk3D<float> Disk3D;
	typedef TSphere3D<float> Sphere3D;
	typedef TCapsule3D<float> Capsule3D;
	typedef TTet3D<float> Tet3D;
	typedef TAlignedBox3D<float> AlignedBox3D;
	typedef TOrientedBox3D<float> OrientedBox3D;
	typedef TCylinder3D<float> Cylinder3D;
	typedef TCone3D<float> Cone3D;
	typedef TMedialCone3D<float> MedialCone3D;
	typedef TMedialSlab3D<float> MedialSlab3D;
#else
	//convenient typedefs 
	typedef TPoint3D<double> Point3D;
	typedef TLine3D<double> Line3D;
	typedef TRay3D<double> Ray3D;
	typedef TSegment3D<double> Segment3D;
	typedef TPlane3D<double> Plane3D;
	typedef TTriangle3D<double> Triangle3D;
	typedef TRectangle3D<double> Rectangle3D;
	typedef TDisk3D<double> Disk3D;
	typedef TSphere3D<double> Sphere3D;
	typedef TCapsule3D<double> Capsule3D;
	typedef TTet3D<double> Tet3D;
	typedef TAlignedBox3D<double> AlignedBox3D;
	typedef TOrientedBox3D<double> OrientedBox3D;
	typedef TCylinder3D<double> Cylinder3D;
	typedef TCone3D<double> Cone3D;
#endif
}

#include "Primitive3D.inl"
