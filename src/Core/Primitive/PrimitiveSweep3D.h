/**
 * @file PrimitiveSweep3D.h
 * @Xiaowei He
 * @brief This class is implemented for continuous collision detection.
 * @version 0.1
 * @date 2020-06-20
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#ifndef PHYSIKA_PRIMITIVE_SWEEP_3D
#define PHYSIKA_PRIMITIVE_SWEEP_3D

#include "Primitive3D.h"

namespace dyno
{
	template <typename Real> class TPointSweep3D;
	template <typename Real> class TTriangleSweep3D;

	template<typename Real>
	class TPointSweep3D
	{
	public:
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;

	public:
		DYN_FUNC TPointSweep3D(TPoint3D<Real>& start, TPoint3D<Real>& end);
		DYN_FUNC TPointSweep3D(const TPointSweep3D& point_sweep);

		/**
		 * @brief Calculate the possible intersection for a moving point and a moving triangle. We assume both the point and the triangle move along a straight line.
		 * 
		 * @param triangle_sweep 
		 * @param t 
		 * @return DYN_FUNC intersect 
		 */
		DYN_FUNC int intersect(const TTriangleSweep3D<Real>& triangle_sweep, typename TTriangle3D<Real>::Param& baryc, Real& t, Real threshold = Real(0.0001)) const;

		/**
		 * @brief Return the intermediate state for a point
		 * 
		 * @param t A paramenter ranging from 0 to 1
		 * @return Intermediate point
		 */
		DYN_FUNC TPoint3D<Real> interpolate(Real t) const;

		TPoint3D<Real> start_point;
		TPoint3D<Real> end_point;
	};



	template<typename Real>
	class TTriangleSweep3D
	{
	public:
		DYN_FUNC TTriangleSweep3D(TTriangle3D<Real>& start, TTriangle3D<Real>& end);
		DYN_FUNC TTriangleSweep3D(const TTriangleSweep3D& triangle_sweep);

		/**
		 * @brief Return the intermediate state for a triangle
		 * 
		 * @param t A paramenter ranging from 0 to 1
		 * @return Intermediate triangle
		 */
		DYN_FUNC TTriangle3D<Real> interpolate(Real t) const;

		TTriangle3D<Real> start_triangle;
		TTriangle3D<Real> end_triangle;
	};

#ifdef PRECISION_FLOAT
	typedef TPointSweep3D<float> PointSweep3D;
	typedef TTriangleSweep3D<float> TriangleSweep3D;
#else
	typedef TPointSweep3D<double> PointSweep3D;
	typedef TTriangleSweep3D<double> TriangleSweep3D;
#endif

}

#include "PrimitiveSweep3D.inl"

#endif

