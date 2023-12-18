/**
 * Copyright 2023 Xiaowei He
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
 * 
 */

#pragma once
#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"

namespace dyno
{
	/**
	 * @brief 0D geometric primitive in two-dimensional space
	 *
	 */
	template <typename Real> class TPoint2D;

	/**
	 * @brief 1D geometric primitives in two-dimensional space
	 *
	 */
	template <typename Real> class TLine2D;
	template <typename Real> class TRay2D;
	template <typename Real> class TSegment2D;
	template <typename Real> class TCircle2D;

	/**
	 * @brief 2D geometric primitives in two-dimensional space
	 *
	 */
	template <typename Real> class TTriangle2D;
	template <typename Real> class TRectangle2D;
	template <typename Real> class TDisk2D;


	template<typename Real>
	class TPoint2D
	{
	public:
		typedef typename Vector<Real, 2> Coord2D;

	public:
		DYN_FUNC TPoint2D();
		DYN_FUNC TPoint2D(const Real& c0, const Real& c1);
		DYN_FUNC TPoint2D(const TPoint2D& pt);

		DYN_FUNC TPoint2D& operator = (const Coord2D& p);

		explicit DYN_FUNC TPoint2D(const Real& val);
		explicit DYN_FUNC TPoint2D(const Coord2D& pos);


		/**
		 * @brief project a point onto linear components -- lines, rays and segments
		 *
		 * @param line/ray/segment linear components
		 * @return closest point
		 */
		DYN_FUNC TPoint2D<Real> project(const TLine2D<Real>& line) const;
		DYN_FUNC TPoint2D<Real> project(const TRay2D<Real>& ray) const;
		DYN_FUNC TPoint2D<Real> project(const TSegment2D<Real>& segment) const;
		DYN_FUNC TPoint2D<Real> project(const TCircle2D<Real>& circle) const;

		DYN_FUNC Real distance(const TPoint2D<Real>& pt) const;
		DYN_FUNC Real distance(const TLine2D<Real>& line) const;
		DYN_FUNC Real distance(const TRay2D<Real>& ray) const;
		DYN_FUNC Real distance(const TSegment2D<Real>& segment) const;
		DYN_FUNC Real distance(const TCircle2D<Real>& circle) const;

		DYN_FUNC Real distanceSquared(const TPoint2D<Real>& pt) const;
		DYN_FUNC Real distanceSquared(const TLine2D<Real>& line) const;
		DYN_FUNC Real distanceSquared(const TRay2D<Real>& ray) const;
		DYN_FUNC Real distanceSquared(const TSegment2D<Real>& segment) const;
		DYN_FUNC Real distanceSquared(const TCircle2D<Real>& circle) const;

		/**
		 * @brief check whether a point strictly lies inside (excluding boundary) a 1D geometric primitive
		 *
		 * @param line/ray/segment 1D geometric primitives
		 * @return true if a point is inside the geometric primitive, otherwise return false
		 */
		DYN_FUNC bool inside(const TLine2D<Real>& line) const;
		DYN_FUNC bool inside(const TRay2D<Real>& ray) const;
		DYN_FUNC bool inside(const TSegment2D<Real>& segment) const;
		DYN_FUNC bool inside(const TCircle2D<Real>& circle) const;

		DYN_FUNC const TSegment2D<Real> operator-(const TPoint2D<Real>& pt) const;

		Coord2D origin;
	};

	template<typename Real>
	class TLine2D
	{
	public:
		typedef typename Vector<Real, 2> Coord2D;

	public:
		DYN_FUNC TLine2D();
		/**
		 * @brief
		 *
		 * @param pos
		 * @param dir = 0 indicate the line degenerates into a point
		 */
		DYN_FUNC TLine2D(const Coord2D& pos, const Coord2D& dir);
		DYN_FUNC TLine2D(const TLine2D<Real>& line);

		DYN_FUNC TSegment2D<Real> proximity(const TLine2D<Real>& line) const;
		DYN_FUNC TSegment2D<Real> proximity(const TRay2D<Real>& ray) const;
		DYN_FUNC TSegment2D<Real> proximity(const TSegment2D<Real>& segment) const;
		DYN_FUNC TSegment2D<Real> proximity(const TCircle2D<Real>& circle) const;

		DYN_FUNC Real distance(const TPoint2D<Real>& pt) const;
		DYN_FUNC Real distance(const TLine2D<Real>& line) const;
		DYN_FUNC Real distance(const TRay2D<Real>& ray) const;
		DYN_FUNC Real distance(const TSegment2D<Real>& segment) const;

		DYN_FUNC Real distanceSquared(const TPoint2D<Real>& pt) const;
		DYN_FUNC Real distanceSquared(const TLine2D<Real>& line) const;
		DYN_FUNC Real distanceSquared(const TRay2D<Real>& ray) const;
		DYN_FUNC Real distanceSquared(const TSegment2D<Real>& segment) const;

		DYN_FUNC int intersect(const TCircle2D<Real>& circle, TSegment2D<Real>& interSeg) const;

		DYN_FUNC Real parameter(const Coord2D& pos) const;

		DYN_FUNC bool isValid() const;

		Coord2D origin;

		//direction will be normalized during construction
		Coord2D direction;
	};

	template<typename Real>
	class TRay2D
	{
	public:
		typedef typename Vector<Real, 2> Coord2D;

	public:
		DYN_FUNC TRay2D();

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
		DYN_FUNC TRay2D(const Coord2D& pos, const Coord2D& dir);
		DYN_FUNC TRay2D(const TRay2D<Real>& ray);

		DYN_FUNC TSegment2D<Real> proximity(const TRay2D<Real>& ray) const;
		DYN_FUNC TSegment2D<Real> proximity(const TSegment2D<Real>& segment) const;

		DYN_FUNC Real distance(const TPoint2D<Real>& pt) const;
		DYN_FUNC Real distance(const TSegment2D<Real>& segment) const;

		DYN_FUNC Real distanceSquared(const TPoint2D<Real>& pt) const;
		DYN_FUNC Real distanceSquared(const TSegment2D<Real>& segment) const;

		DYN_FUNC int intersect(const TCircle2D<Real>& sphere, TSegment2D<Real>& interSeg) const;

		DYN_FUNC Real parameter(const Coord2D& pos) const;

		DYN_FUNC bool isValid() const;

		Coord2D origin;

		//guarantee direction is a unit vector
		Coord2D direction;
	};

	template<typename Real>
	class TSegment2D
	{
	public:
		typedef typename Vector<Real, 2> Coord2D;

	public:
		DYN_FUNC TSegment2D();
		DYN_FUNC TSegment2D(const Coord2D& p0, const Coord2D& p1);
		DYN_FUNC TSegment2D(const TSegment2D<Real>& segment);

		DYN_FUNC TSegment2D<Real> proximity(const TSegment2D<Real>& segment) const;

		DYN_FUNC Real distance(const TSegment2D<Real>& segment) const;

		DYN_FUNC Real distanceSquared(const TSegment2D<Real>& segment) const;

		DYN_FUNC Real length() const;
		DYN_FUNC Real lengthSquared() const;

		DYN_FUNC int intersect(const TCircle2D<Real>& circle, TSegment2D<Real>& interSeg) const;

		DYN_FUNC Real parameter(const Coord2D& pos) const;

		inline DYN_FUNC Coord2D& startPoint() { return v0; }
		inline DYN_FUNC Coord2D& endPoint() { return v1; }

		inline DYN_FUNC Coord2D startPoint() const { return v0; }
		inline DYN_FUNC Coord2D endPoint() const { return v1; }

		inline DYN_FUNC Coord2D direction() const { return v1 - v0; }

		inline DYN_FUNC TSegment2D<Real> operator-(void) const;

		DYN_FUNC bool isValid() const;

		Coord2D v0;
		Coord2D v1;
	};

	template<typename Real>
	class TCircle2D
	{
	public:
		typedef typename Vector<Real, 2> Coord2D;

	public:
		DYN_FUNC TCircle2D();
		DYN_FUNC TCircle2D(const Coord2D& c, const Real& r);
		DYN_FUNC TCircle2D(const TCircle2D<Real>& circle);

		Coord2D center;
		Real radius;
		Real theta;
	};

	template<typename Real>
	class TAlignedBox2D
	{
	public:
		typedef typename Vector<Real, 2> Coord2D;

	public:
		DYN_FUNC TAlignedBox2D();
		DYN_FUNC TAlignedBox2D(const Coord2D& p0, const Coord2D& p1);
		DYN_FUNC TAlignedBox2D(const TAlignedBox2D<Real>& box);


		Coord2D v0;
		Coord2D v1;
	};

#define MAX_POLYGON_VERTEX_NUM 8

	template<typename Real>
	class TPolygon2D
	{
	public:
		typedef typename Vector<Real, 2> Coord2D;

	public:
		TPolygon2D();
		~TPolygon2D();

		void setAsBox(Real hx, Real hy);

		void setAsPentagon(const Coord2D& v0, const Coord2D& v1, const Coord2D& v2, const Coord2D& v3, const Coord2D& v4);

		void setAsTriangle(const Coord2D& v0, const Coord2D& v1, const Coord2D& v2);

		void setAsLine(const Coord2D& v0, const Coord2D& v1);

		const uint vertexSize() const { return size; };

		inline const Coord2D& vertex(uint i) const { return _vertices[i]; }
		inline const Coord2D& normal(uint i) const { return _normals[i]; }
		inline const Coord2D& center() const { return _center; }

		inline void setCenter(const Coord2D& c) { _center = c; }
		inline void setVertex(const uint i, const Coord2D& v) { _vertices[i] = v; }

		inline Real radius() const { return _radius; }

		TAlignedBox2D<Real> aabb();

	private:
		Coord2D _center;
		Coord2D _vertices[MAX_POLYGON_VERTEX_NUM];
		Coord2D _normals[MAX_POLYGON_VERTEX_NUM];

		uint size = 0;
		Real _radius = 0.005f;
	};

	template class TAlignedBox2D<float>;
	template class TAlignedBox2D<double>;
}

#include "Primitive2D.inl"
