//#include "Primitive3D.h"
#include "Complex.h"

#include "CCD/TightCCD.h"


namespace dyno
{
	template<typename Real>
	DYN_FUNC TPointSweep3D<Real>::TPointSweep3D(TPoint3D<Real>& start, TPoint3D<Real>& end)
	{
		start_point = start;
		end_point = end;
	}


	template<typename Real>
	DYN_FUNC TPointSweep3D<Real>::TPointSweep3D(const TPointSweep3D& point_sweep)
	{
		start_point = point_sweep.start_point;
		end_point = point_sweep.end_point;
	}

	template<typename Real>
	DYN_FUNC bool TPointSweep3D<Real>::intersect(const TTriangleSweep3D<Real>& triangle_sweep, typename TTriangle3D<Real>::Param& baryc, Real& t, const Real threshold) const
	{
		bool collided = TightCCD<Real>::VertexFaceCCD(
			start_point.origin, triangle_sweep.start_triangle.v[0], triangle_sweep.start_triangle.v[1], triangle_sweep.start_triangle.v[2],
			end_point.origin, triangle_sweep.end_triangle.v[0], triangle_sweep.end_triangle.v[1], triangle_sweep.end_triangle.v[2],
			t);

		//check whether the intersection point lies inside the triangle
		if (collided)
		{
			Point3D p_ret = interpolate(t);
			Triangle3D tri_ret = triangle_sweep.interpolate(t);

			typename TTriangle3D<Real>::Param baryParam;
			bool bValid = tri_ret.computeBarycentrics(p_ret.origin, baryParam);
			Real dist = glm::abs(p_ret.distance(tri_ret));
			bool bIntersected = baryParam.u >= Real(0) & baryParam.u <= Real(1) + threshold & baryParam.v >= Real(0) - threshold & baryParam.v <= Real(1) + threshold & baryParam.w >= Real(0) - threshold & baryParam.w <= Real(1) + threshold;
			if (bValid && bIntersected)
			{

				baryc.u = max(Real(0), min(baryParam.u, Real(1)));
				baryc.v = max(Real(0), min(baryParam.v, Real(1)));
				baryc.w = max(Real(0), min(baryParam.w, Real(1)));

				return true;
			}
			else
				return false;
		}

		return true;
	}

	template<typename Real>
	DYN_FUNC TPoint3D<Real> TPointSweep3D<Real>::interpolate(Real t) const
	{
		TPoint3D<Real> point;
		point.origin = (1 - t)*start_point.origin + t * end_point.origin;

		return point;
	}


	template<typename Real>
	DYN_FUNC TTriangleSweep3D<Real>::TTriangleSweep3D(TTriangle3D<Real>& start, TTriangle3D<Real>& end)
	{
		start_triangle = start;
		end_triangle = end;
	}


	template<typename Real>
	DYN_FUNC TTriangleSweep3D<Real>::TTriangleSweep3D(const TTriangleSweep3D& triangle_sweep)
	{
		start_triangle = triangle_sweep.start_triangle;
		end_triangle = triangle_sweep.end_triangle;
	}


	template<typename Real>
	DYN_FUNC TTriangle3D<Real> TTriangleSweep3D<Real>::interpolate(Real t) const
	{
		Real t0 = t;
		Real t1 = 1 - t;

		TTriangle3D<Real> triangle;
		triangle.v[0] = t1 * start_triangle.v[0] + t0 * end_triangle.v[0];
		triangle.v[1] = t1 * start_triangle.v[1] + t0 * end_triangle.v[1];
		triangle.v[2] = t1 * start_triangle.v[2] + t0 * end_triangle.v[2];

		return triangle;
	}

}
