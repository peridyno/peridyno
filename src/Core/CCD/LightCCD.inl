#define CCD_DEBUG_PRINT 0
#if CCD_DEBUG_PRINT
#define CCD_PRINT(...) printf(__VA_ARGS__)
#else
#define CCD_PRINT(...)
#endif
#include "Vector.h"
#include "Math/SimpleMath.h"

namespace dyno
{
	template<typename T>
	DYN_FUNC bool LightCCD<T>::VertexFaceCCD(
		const Vector<T, 3>& p0, const Vector<T, 3>& a0, const Vector<T, 3>& b0, const Vector<T, 3>& c0,
		const Vector<T, 3>& p1, const Vector<T, 3>& a1, const Vector<T, 3>& b1, const Vector<T, 3>& c1,
		T& time, T pid, T tid)
	{
		Vector<T, 3> AB = p1 - p0;//-V_point
		Vector<T, 3> CD = a1 - a0;//-V_triangle
		Vector<T, 3> BD = p0 - a0;
		CCD_PRINT("[CCD] AB=(%.8f, %.8f, %.8f),CD=(%.8f, %.8f, %.8f),BD=(%.8f, %.8f, %.8f)\n",
			AB.x, AB.y, AB.z,
			CD.x, CD.y, CD.z,
			BD.x, BD.y, BD.z);
		TTriangle3D<Real> t0(a0, b0, c0);
		TTriangle3D<Real> t1(a1, b1, c1);

		Vector<T, 3> n0 = t0.normal();
		Vector<T, 3> n1 = t1.normal();

		Vector<T, 3> n10 = n1 - n0;
		CCD_PRINT("[CCD] n0=(%.8f, %.8f, %.8f), n1=(%.8f, %.8f, %.8f), n10=(%.8f, %.8f, %.8f)\n",
			n0.x, n0.y, n0.z,
			n1.x, n1.y, n1.z,
			n10.x, n10.y, n10.z);
		Real a = (AB - CD).dot(n10);
		Real b = BD.dot(n10) + (AB - CD).dot(n0);
		CCD_PRINT("[CCD] (AB - CD)=(%.8f, %.8f, %.8f),(AB - CD).dot(n0)=%.8f\n",
			(AB - CD).x, (AB - CD).y, (AB - CD).z, (AB - CD).dot(n0));
		Real c = BD.dot(n0);
		CCD_PRINT("[CCD] a=%.8f, b=%.8f, c=%.8f\n", a, b, c);
		if (glm::abs(a) < EPSILON)
		{
			if (glm::abs(b) < EPSILON)
			{
				if (glm::abs(c) < EPSILON)
				{
					TTriangle3D<T> t_fix(a0, b0, c0);
					typename TTriangle3D<Real>::Param tParam;
					t_fix.computeBarycentrics(p0, tParam);
					Real u0 = tParam.u;
					Real v0 = tParam.v;
					Real w0 = tParam.w;
					CCD_PRINT("[CCD] barycentric at p0: u0=%.8f, v0=%.8f, w0=%.8f\n", u0, v0, w0);
					bool inside = u0 >= (-EPSILON) && u0 <= Real(1) && v0 >= (-EPSILON) && v0 <= Real(1) && w0 >= (-EPSILON) && w0 <= Real(1);
					if (inside)
					{
						CCD_PRINT("[CCD] p0 inside triangle at t=0, return toi=0\n");
						time = 0;
						return true;
					}
					Real min_time;
					Real max_time;
					typename TTriangle3D<Real>::Param tParam_v;
					Vector<T, 3> V_rel = -AB - (-CD);

					typename TTriangle3D<Real>::Param tParam_end;
					Vector<T, 3> p_end = p0 + V_rel;
					t_fix.computeBarycentrics(p_end, tParam_end);
					Real u1 = tParam_end.u;
					Real v1 = tParam_end.v;
					Real w1 = tParam_end.w;
					CCD_PRINT("[CCD] barycentric at p_end: u1=%.8f, v1=%.8f, w1=%.8f\n", u1, v1, w1);
					Real j = u1 - u0;
					Real k = v1 - v0;
					Real l = j - k;
					if (glm::abs(j) < EPSILON || glm::abs(k) < EPSILON || glm::abs(l) < EPSILON)
					{
						time = 1;
						return false;
					}
					if (j > 0 && k > 0)
					{
						if (l > 0)
						{
							min_time = maximum(Real(0), maximum(-u0 / j, -v0 / k));
							max_time = minimum(Real(1), w0 / l);
						}
						else
						{
							min_time = maximum(Real(0), maximum(maximum(-u0 / j, -v0 / k), w0 / l));
							max_time = Real(1);
						}
					}
					else if (j < 0 && k > 0)
					{
						min_time = maximum(maximum(-v0 / k, w0 / l), Real(0));
						max_time = minimum(-u0 / j, Real(1));

					}
					else if (j > 0 && k < 0)
					{
						min_time = maximum(Real(0), -u0 / j);
						max_time = minimum(minimum(w0 / l, -v0 / k), Real(1));
					}
					else
					{
						if (l > 0)
						{
							min_time = Real(0);
							max_time = minimum(Real(1), minimum(minimum(-u0 / j, -v0 / k), w0 / l));
						}
						else
						{
							min_time = maximum(Real(0), w0 / l);
							max_time = minimum(Real(1), minimum(-u0 / j, -v0 / k));
						}
					}
					time = min_time <= max_time ? min_time : -1;
					CCD_PRINT("[CCD] min_time=%.8f, max_time=%.8f, chosen time=%.8f\n", min_time, max_time, time);
					u0 += time * u1;
					v0 += time * v1;
					w0 = 1 - u0 - v0;
					CCD_PRINT("[CCD] barycentric at toi: u=%.8f, v=%.8f, w=%.8f\n", u0, v0, w0);
					inside = u0 >= (-EPSILON) && u0 <= Real(1) && v0 >= (-EPSILON) && v0 <= Real(1) && w0 >= (-EPSILON) && w0 <= Real(1);
					if ((time <= 1) && (time >= 0) && inside)
						CCD_PRINT("[CCD] toi in [0,1] and inside triangle, return true\n");
					else
						CCD_PRINT("[CCD] toi not valid or not inside triangle, return false\n");
					return (time <= 1) && (time >= 0);
				}
				else {
					CCD_PRINT("[CCD] c not near zero, return false\n");
					return false;
				}
			}
			else
			{
				time = -c / b;
				//time = 1 - time;
				CCD_PRINT("[CCD] linear case, time=%.8f\n", time);
			}
		}
		else
		{
			Real delta = b * b - 4 * a * c;
			CCD_PRINT("[CCD] quadratic case, delta=%.8f\n", delta);
			if (delta < 0)
			{
				CCD_PRINT("[CCD] delta < 0, no real root, return false\n");
				time = 1;
				return false;
			}
			else if (delta == 0)
			{
				time = -b / (2 * a);
				//time = 1 - time;
				CCD_PRINT("[CCD] delta=0, time=%.8f\n", time);
			}
			else
			{
				Real t0 = (-b - glm::sqrt(delta)) / (2 * a);
				Real t1 = (-b + glm::sqrt(delta)) / (2 * a);
				Real t_check = t0 >= 0 ? t0 : t1;
				time = t_check <= 1 ? t_check : 1;
				//time = 1 - time;
				CCD_PRINT("[CCD] quadratic roots: t0=%.8f, t1=%.8f, chosen time=%.8f\n", t0, t1, time);
			}
		}

		//Check when the intersection point is located inside the triangle
		Vector<T, 3> v0 = a0 + time * (a1 - a0);
		Vector<T, 3> v1 = b0 + time * (b1 - b0);
		Vector<T, 3> v2 = c0 + time * (c1 - c0);

		Vector<T, 3> p_hit = p0 + time * (p1 - p0);
		TTriangle3D<T> t_hit(v0, v1, v2);
		CCD_PRINT("[CCD] v0=(%.8f, %.8f, %.8f)\n", v0.x, v0.y, v0.z);
		CCD_PRINT("[CCD] v1=(%.8f, %.8f, %.8f)\n", v1.x, v1.y, v1.z);
		CCD_PRINT("[CCD] v2=(%.8f, %.8f, %.8f)\n", v2.x, v2.y, v2.z);
		CCD_PRINT("[CCD] p_hit=(%.8f, %.8f, %.8f)\n", p_hit.x, p_hit.y, p_hit.z);

		typename TTriangle3D<Real>::Param tParam;
		t_hit.computeBarycentrics(p_hit, tParam);
		CCD_PRINT("[CCD] barycentric at hit: u=%.8f, v=%.8f, w=%.8f\n", tParam.u, tParam.v, tParam.w);

		bool inside = tParam.u >= (-EPSILON * 10000) && tParam.u <= Real(1 + EPSILON * 10000) && tParam.v >= (-EPSILON * 10000) && tParam.v <= Real(1 + EPSILON * 10000) && tParam.w >= (-EPSILON * 10000) && tParam.w <= Real(1 + EPSILON * 10000);
		if (inside && (time < 1 + EPSILON) && (time > -EPSILON))
			CCD_PRINT("[CCD] toi=%.8f, inside triangle at hit, return true\n", time);
		else
			CCD_PRINT("[CCD] toi=%.8f, not inside triangle or out of [0,1], return false\n", time);
		return inside && (time < 1) && (time > 0);
	}
}