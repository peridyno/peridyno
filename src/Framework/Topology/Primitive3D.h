#ifndef PHYSIKA_PRIMITIVE_3D
#define PHYSIKA_PRIMITIVE_3D
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

namespace dyno
{
	// #ifdef PRECISION_FLOAT
	// 	#define REAL_EPSILON 1e-5
	// 	#define  REAL_EPSILON_SQUARED 1e-10
	// #else
	// 	#define REAL_EPSILON 1e-10
	// 	#define  REAL_EPSILON_SQUARED 1e-20
	// #endif

#ifdef PRECISION_FLOAT
	typedef Vec2f Coord2D;
	typedef Vec3f Coord3D;
	typedef Mat3f Matrix3D;
#else
	typedef Vec2d Coord2D;
	typedef Vec3d Coord3D;
	typedef Mat3d Matrix3D;
#endif

	constexpr Real REAL_EPSILON = (std::numeric_limits<Real>::epsilon)();
	constexpr Real REAL_EPSILON_SQUARED = REAL_EPSILON * REAL_EPSILON;


#ifdef PRECISION_FLOAT
	typedef Vec2f Coord2D;
	typedef Vec3f Coord3D;
	typedef Mat3f Matrix3D;
#else
	typedef Vec2d Coord2D;
	typedef Vec3d Coord3D;
	typedef Mat3d Matrix3D;
#endif

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

	template<typename Real>
	class TPoint3D
	{
	public:
		DYN_FUNC TPoint3D();
		DYN_FUNC TPoint3D(const Real& c0, const Real& c1, const Real& c2);
		DYN_FUNC TPoint3D(const TPoint3D& pt);

		DYN_FUNC TPoint3D operator = (const Coord3D& p);

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


		DYN_FUNC int intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const;
		DYN_FUNC int intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const;

		DYN_FUNC int intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const;
		DYN_FUNC int intersect(const TTet3D<Real>& tet, TSegment3D<Real>& interSeg) const;
		DYN_FUNC int intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const;


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
		DYN_FUNC TSegment3D();
		DYN_FUNC TSegment3D(const Coord3D& p0, const Coord3D& p1);
		DYN_FUNC TSegment3D(const TSegment3D<Real>& segment);

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
		DYN_FUNC TSphere3D();
		DYN_FUNC TSphere3D(const Coord3D& c, const Real& r);
		DYN_FUNC TSphere3D(const TSphere3D<Real>& sphere);

		DYN_FUNC Real volume();

		DYN_FUNC bool isValid();

		DYN_FUNC TAlignedBox3D<Real> aabb();

		Real radius;
		Coord3D center;
	};

	template<typename Real>
	class TCapsule3D
	{
	public:
		DYN_FUNC TCapsule3D();
		DYN_FUNC TCapsule3D(const Coord3D& v0, const Coord3D& v1, const Real& r);
		DYN_FUNC TCapsule3D(const TCapsule3D<Real>& capsule);

		DYN_FUNC Real volume();

		DYN_FUNC bool isValid();
		DYN_FUNC TAlignedBox3D<Real> aabb();

		Real radius;
		TSegment3D<Real> segment;
	};

	/**
	 * @brief vertices are ordered so that the normal vectors for the triangular faces point outwards
	 *
	 */
	template<typename Real>
	class TTet3D
	{
	public:
		DYN_FUNC TTet3D();
		DYN_FUNC TTet3D(const Coord3D& v0, const Coord3D& v1, const Coord3D& v2, const Coord3D& v3);
		DYN_FUNC TTet3D(const TTet3D<Real>& tet);

		DYN_FUNC TTriangle3D<Real> face(const int index) const;

		DYN_FUNC Real volume() const;

		DYN_FUNC bool isValid();

		//DYN_FUNC bool intersect(const TTet3D<Real>& tet, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance = 1) const;
		DYN_FUNC bool intersect(const TTet3D<Real>& tet, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance = 1) const;
		DYN_FUNC bool intersect(const TTriangle3D<Real>& tri, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance = 1) const;

		DYN_FUNC void expand(Real r);
		DYN_FUNC TAlignedBox3D<Real> aabb();

		// http://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter
		DYN_FUNC TPoint3D<Real> circumcenter() const;
		DYN_FUNC TPoint3D<Real> barycenter() const;

		Coord3D v[4];
	};

	template<typename Real>
	class TAlignedBox3D
	{
	public:
		DYN_FUNC TAlignedBox3D();
		DYN_FUNC TAlignedBox3D(const Coord3D& p0, const Coord3D& p1);
		DYN_FUNC TAlignedBox3D(const TAlignedBox3D<Real>& box);

		DYN_FUNC bool intersect(const TAlignedBox3D<Real>& abox, TAlignedBox3D<Real>& interBox) const;
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

#ifdef PRECISION_FLOAT
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
#else
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
#endif

}

#include "Primitive3D.inl"

#endif

