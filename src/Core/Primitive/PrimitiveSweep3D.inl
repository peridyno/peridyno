//#include "Primitive3D.h"
#include "Complex.h"


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
	DYN_FUNC int TPointSweep3D<Real>::intersect(const TTriangleSweep3D<Real>& triangle_sweep, typename TTriangle3D<Real>::Param& baryc, Real& t, Real threshold) const
	{
		t = -Real(1);
		Real lowLimit = -Real(5)*REAL_EPSILON;
		Real highLimit = Real(1) + Real(5)*REAL_EPSILON;
		//Interval<Real> tRange(lowLimit, highLimit);
		
		Real A00, A01, A02;
		Real A10, A11, A12;
		Real A20, A21, A22;

		Real B00, B01, B02;
		Real B10, B11, B12;
		Real B20, B21, B22;

		Coord3D p = start_point.origin;
		Coord3D q = end_point.origin - start_point.origin;

		Coord3D v0 = triangle_sweep.start_triangle.v[0];
		Coord3D v1 = triangle_sweep.start_triangle.v[1];
		Coord3D v2 = triangle_sweep.start_triangle.v[2];
		Coord3D l = triangle_sweep.end_triangle.v[0] - triangle_sweep.start_triangle.v[0];
		Coord3D m = triangle_sweep.end_triangle.v[1] - triangle_sweep.start_triangle.v[1];
		Coord3D n = triangle_sweep.end_triangle.v[2] - triangle_sweep.start_triangle.v[2];

		A00 = q[0] - l[0];	A01 = q[1] - l[1];	A02 = q[2] - l[2];
		A10 = m[0] - l[0];	A11 = m[1] - l[1];	A12 = m[2] - l[2];
		A20 = n[0] - l[0];	A21 = n[1] - l[1];	A22 = n[2] - l[2];

		B00 = p[0] - v0[0];	B01 = p[1] - v0[1];	B02 = p[2] - v0[2];
		B10 = v1[0] - v0[0];	B11 = v1[1] - v0[1];	B12 = v1[2] - v0[2];
		B20 = v2[0] - v0[0];	B21 = v2[1] - v0[1];	B22 = v2[2] - v0[2];

		//solving determinant|A*t+B| = 0
		//determinant|A*t+B| = -A02 * A11*A20*t ^ 3 + A01 * A12*A20*t ^ 3 + A02 * A10*A21*t ^ 3 - A00 * A12*A21*t ^ 3 - A01 * A10*A22*t ^ 3 + A00 * A11*A22*t ^ 3 - A12 * A21*B00*t ^ 2 + A11 * A22*B00*t ^ 2 + A12 * A20*B01*t ^ 2 - A10 * A22*B01*t ^ 2 - A11 * A20*B02*t ^ 2 + A10 * A21*B02*t ^ 2 + A02 * A21*B10*t ^ 2 - A01 * A22*B10*t ^ 2 - A02 * A20*B11*t ^ 2 + A00 * A22*B11*t ^ 2 + A01 * A20*B12*t ^ 2 - A00 * A21*B12*t ^ 2 - A02 * A11*B20*t ^ 2 + A01 * A12*B20*t ^ 2 + A02 * A10*B21*t ^ 2 - A00 * A12*B21*t ^ 2 - A01 * A10*B22*t ^ 2 + A00 * A11*B22*t ^ 2 - A22 * B01*B10*t + A21 * B02*B10*t + A22 * B00*B11*t - A20 * B02*B11*t - A21 * B00*B12*t + A20 * B01*B12*t + A12 * B01*B20*t - A11 * B02*B20*t - A02 * B11*B20*t + A01 * B12*B20*t - A12 * B00*B21*t + A10 * B02*B21*t + A02 * B10*B21*t - A00 * B12*B21*t + A11 * B00*B22*t - A10 * B01*B22*t - A01 * B10*B22*t + A00 * B11*B22*t - B02 * B11*B20 + B01 * B12*B20 + B02 * B10*B21 - B00 * B12*B21 - B01 * B10*B22 + B00 * B11*B22
		Real a = -A02 * A11*A20 + A01 * A12*A20 + A02 * A10*A21 - A00 * A12*A21 - A01 * A10*A22 + A00 * A11*A22;
		Real b = -A12 * A21*B00 + A11 * A22*B00 + A12 * A20*B01 - A10 * A22*B01 - A11 * A20*B02 + A10 * A21*B02 + A02 * A21*B10 - A01 * A22*B10 - A02 * A20*B11 + A00 * A22*B11 + A01 * A20*B12 - A00 * A21*B12 - A02 * A11*B20 + A01 * A12*B20 + A02 * A10*B21 - A00 * A12*B21 - A01 * A10*B22 + A00 * A11*B22;
		Real c = -A22 * B01*B10 + A21 * B02*B10 + A22 * B00*B11 - A20 * B02*B11 - A21 * B00*B12 + A20 * B01*B12 + A12 * B01*B20 - A11 * B02*B20 - A02 * B11*B20 + A01 * B12*B20 - A12 * B00*B21 + A10 * B02*B21 + A02 * B10*B21 - A00 * B12*B21 + A11 * B00*B22 - A10 * B01*B22 - A01 * B10*B22 + A00 * B11*B22;
		Real d = -B02 * B11*B20 + B01 * B12*B20 + B02 * B10*B21 - B00 * B12*B21 - B01 * B10*B22 + B00 * B11*B22;
		
		Real max_length = max(max(l.norm(), m.norm()), n.norm());
		Real max_length3 = max_length * max_length*max_length;

// 		a = 1;
// 		b = 0;
// 		c = 0;
// 		d = -1;

		{
			TPoint3D<Real> proj_ed = end_point.project(triangle_sweep.end_triangle);
			TPoint3D<Real> proj_st = start_point.project(triangle_sweep.start_triangle);

			Real dist_ed = (proj_ed.origin - end_point.origin).norm();
			Real dist_st = (proj_st.origin - start_point.origin).norm();

			Real dist = (proj_st.origin - proj_ed.origin).norm();

			if (dist_ed <= threshold)
			{
				//if (dist_ed < dist_st)
				triangle_sweep.end_triangle.computeBarycentrics(proj_ed.origin, baryc);
				t = dist > REAL_EPSILON ? dist_ed / dist : Real(1);


				return 1;
			}
		}

		
		int num = 0;
		
		if (glm::abs(a) > REAL_EPSILON*max_length3)
		{
			Real a2 = a * a;
			Real a3 = a2 * a;

			Real b2 = b * b;
			Real b3 = b2 * b;

			Real c2 = c * c;
			Real c3 = c2 * c;

			Real d2 = d * d;
			Real d3 = d2 * d;


			Complex<Real> ei(-1 / Real(3) * b2 * c2 + 4 / Real(3) * a*c3 + 9 * a2 * d2 + 2 / Real(3) * (2 * b3 - 9 * a*b*c)*d, Real(0));

			Real f = -1 / Real(27) * b3 / a3 + 1 / Real(6) * b*c / a2 - 1 / Real(2) * d / a;
			Real g = b2 / a2 - 3 * c / a;

			Complex<Real> h = pow(f + 1 / Real(6) * sqrt(ei) / a2, Real(1) / Real(3));

			Complex<Real> w(Real(1), glm::sqrt(Real(3)));
			Complex<Real> w2(Real(1), -glm::sqrt(Real(3)));

			
			Complex<Real> t1 = -1 / Real(18) * w2*g / h - 1 / Real(2) * w*h - 1 / Real(3) * b / a;
			Complex<Real> t2 = -1 / Real(18) * w*g / h - 1 / Real(2) * w2*h - 1 / Real(3) * b / a;
			Complex<Real> t3 = 1 / Real(9) * g / h + h - 1 / Real(3) * b / a;

			if (t1.isReal() && (lowLimit <= t1.realPart() && t1.realPart() <= highLimit))//tRange.inside(t1.realPart()))
			{
				t = t1.realPart();
				num = 1;
			}
			if (t2.isReal() && (lowLimit <= t1.realPart() && t1.realPart() <= highLimit))//tRange.inside(t2.realPart()))
			{
				t = num > 0 ? min(t, t2.realPart()) : t2.realPart();
				num = 1;
			}
			if (t3.isReal() && (lowLimit <= t1.realPart() && t1.realPart() <= highLimit))//tRange.inside(t3.realPart()))
			{
				t = num > 0 ? min(t, t3.realPart()) : t3.realPart();
				num = 1;
			}
			//return num;
		}
		else
		{
			if (glm::abs(b) > REAL_EPSILON*max_length3)
			{
				Real e = c * c - 4 * b*d;
				if (e >= 0)
				{
					Real t1 = (-c + glm::sqrt(e)) / (2 * b);
					Real t2 = (-c - glm::sqrt(e)) / (2 * b);

					if (t1 >= -REAL_EPSILON && t1 <= 1+ REAL_EPSILON)
					{
						t = t1;
						num = 1;
					}

					if (t2 >= 0 && t2 <= 1)
					{
						t = num > 0 ? min(t, t2) : t2;
						num = 1;
					}
				}

			}
			else if (glm::abs(c) > REAL_EPSILON*max_length3)
			{
				Real t1 = -d / c;
				if (t1 >= -REAL_EPSILON && t1 <= 1 + REAL_EPSILON)
				{
					t = t1;
					num = 1;
				}
			}
		}

		//check whether the intersection point lies inside the triangle
		if (num == 1)
		{
			Point3D p_ret = interpolate(t);
			Triangle3D tri_ret = triangle_sweep.interpolate(t);

			typename TTriangle3D<Real>::Param baryParam;
			bool bValid = tri_ret.computeBarycentrics(p_ret.origin, baryParam);
			Real dist = glm::abs(p_ret.distance(tri_ret));
			bool bIntersected = baryParam.u >= Real(0) - threshold & baryParam.u <= Real(1) + threshold & baryParam.v >= Real(0) - threshold & baryParam.v <= Real(1) + threshold & baryParam.w >= Real(0) - threshold & baryParam.w <= Real(1) + threshold;
			if (bValid && bIntersected)
			{
				
				baryc.u = max(Real(0), min(baryParam.u, Real(1)));
				baryc.v = max(Real(0), min(baryParam.v, Real(1)));
				baryc.w = max(Real(0), min(baryParam.w, Real(1)));
			}
			else
				num = 0;
		}
		

		return num;
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
