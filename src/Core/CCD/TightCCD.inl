#include "Vector.h"

namespace dyno
{
	#define REAL_infinity 1.0e30
	#define	REAL_EQUAL(a,b)  (((a < b + EPSILON) && (a > b - EPSILON)) ? true : false)


	template<typename T>
	inline DYN_FUNC T STP(
		const Vector<T, 3>& u, const Vector<T, 3>& v, const Vector<T, 3>& w)
	{
		return u.dot(v.cross(w));
	}

	template<typename T>
	inline DYN_FUNC void fswap(T& a, T& b)
	{
		T t = b;
		b = a;
		a = t;
	}

	template<typename T>
	DYN_FUNC T SignedDistanceVF(
		const Vector<T, 3>& x,
		const Vector<T, 3>& y0,
		const Vector<T, 3>& y1,
		const Vector<T, 3>& y2,
		Vector<T, 3>* n,
		T* w)
	{
		Vector<T, 3> _n; if (!n) n = &_n;
		T _w[4]; if (!w) w = _w;
		Vector<T, 3> y01 = y1 - y0;	y01.normalize();
		Vector<T, 3> y02 = y2 - y0;	y02.normalize();

		*n = y01.cross(y02);
		if ((*n).normSquared() < 1e-6)
			return REAL_infinity;
		*n = (*n).normalize();
		T h = (x - y0).dot(*n);
		T b0 = STP(y1 - x, y2 - x, *n),
			b1 = STP(y2 - x, y0 - x, *n),
			b2 = STP(y0 - x, y1 - x, *n);
		w[0] = 1;
		w[1] = -b0 / (b0 + b1 + b2);
		w[2] = -b1 / (b0 + b1 + b2);
		w[3] = -b2 / (b0 + b1 + b2);
		return h;
	}

	template<typename T>
	DYN_FUNC T SignedDistanceEE(
		const Vector<T, 3>& x0, const Vector<T, 3>& x1,
		const Vector<T, 3>& y0, const Vector<T, 3>& y1,
		Vector<T, 3>* n, 
		Real* w) 
	{
		Vector<T, 3> _n; if (!n) n = &_n;
		T _w[4]; if (!w) w = _w;

		Vector<T, 3> x01 = (x1 - x0);		x01.normalize();
		Vector<T, 3> y01 = (y1 - y0);		y01.normalize();

		*n = x01.cross(y01);
		if ((*n).normSquared() < 1e-6)
			return REAL_infinity;

		*n = (*n).normalize();
		T h = (x0 - y0).dot(*n);
		T a0 = STP(y1 - x1, y0 - x1, *n);
		T a1 = STP(y0 - x0, y1 - x0, *n);
		T b0 = STP(x0 - y1, x1 - y1, *n);
		T b1 = STP(x1 - y0, x0 - y0, *n);
		w[0] = a0 / (a0 + a1);
		w[1] = a1 / (a0 + a1);
		w[2] = -b0 / (b0 + b1);
		w[3] = -b1 / (b0 + b1);
		return h;
	}

	template<typename T>
	DYN_FUNC T triProduct(Vector<T, 3>& a, Vector<T, 3>& b, Vector<T, 3>& c)
	{
		return a.cross(b).dot(c);
	}

	template<typename T>
	DYN_FUNC Vector<T, 3> xvpos(Vector<T, 3> x, Vector<T, 3> v, T t)
	{
		return x + v * t;
	}

	template<typename T>
	DYN_FUNC int sgn(T x) { return x < 0 ? -1 : 1; }

	template<typename T>
	inline DYN_FUNC int SolveQuadratic(T a, T b, T c, T x[2]) {
		// http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
		T d = b * b - 4 * a * c;
		if (d < 0) {
			x[0] = -b / (2 * a);
			return 0;
		}
		T q = -(b + sgn(b) * sqrt(d)) / 2;
		int i = 0;
		if (abs(a) > 1e-12 * abs(q))
			x[i++] = q / a;
		if (abs(q) > 1e-12 * abs(c))
			x[i++] = c / q;
		if (i == 2 && x[0] > x[1])
			fswap(x[0], x[1]);
		return i;
	}

	template<typename T>
	inline DYN_FUNC T NewtonsMethod(T a, T b, T c, T d, T x0, int init_dir) 
	{
		if (init_dir != 0) {
			// quadratic approximation around x0, assuming y' = 0
			T y0 = d + x0 * (c + x0 * (b + x0 * a)),
				ddy0 = 2 * b + x0 * (6 * a);
			x0 += init_dir * sqrt(abs(2 * y0 / ddy0));
		}
		for (int iter = 0; iter < 100; iter++) {
			T y = d + x0 * (c + x0 * (b + x0 * a));
			T dy = c + x0 * (2 * b + x0 * 3 * a);
			if (dy == 0)
				return x0;
			T x1 = x0 - y / dy;
			if (abs(x0 - x1) < 1e-6)
				return x0;
			x0 = x1;
		}
		return x0;
	}

	// solves a x^3 + b x^2 + c x + d == 0
	template<typename T>
	inline DYN_FUNC	int SolveCubic(T a, T b, T c, T d, T x[3])
	{
		T xc[2];
		int ncrit = SolveQuadratic(3 * a, 2 * b, c, xc);
		if (ncrit == 0) {
			x[0] = NewtonsMethod(a, b, c, d, xc[0], 0);
			return 1;
		}
		else if (ncrit == 1) {// cubic is actually quadratic
			return SolveQuadratic(b, c, d, x);
		}
		else {
			T yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
				d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
			int i = 0;
			if (yc[0] * a >= 0)
				x[i++] = NewtonsMethod(a, b, c, d, xc[0], -1);
			if (yc[0] * yc[1] <= 0) {
				int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
				x[i++] = NewtonsMethod(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
			}
			if (yc[1] * a <= 0)
				x[i++] = NewtonsMethod(a, b, c, d, xc[1], 1);
			return i;
		}
	}

	template<typename T>
	DYN_FUNC bool CollisionTest(
		const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2, const Vector<T, 3>& x3,
		const Vector<T, 3>& v0, const Vector<T, 3>& v1, const Vector<T, 3>& v2, const Vector<T, 3>& v3,
		T& time, const int isVF)//0:vf  1:ee
	{
		T a0 = STP(x1, x2, x3);
		T a1 = STP(v1, x2, x3) + STP(x1, v2, x3) + STP(x1, x2, v3);
		T a2 = STP(x1, v2, v3) + STP(v1, x2, v3) + STP(v1, v2, x3);
		T a3 = STP(v1, v2, v3);

		if (REAL_EQUAL(a1, 0) && REAL_EQUAL(a2, 0) && REAL_EQUAL(a1, 0))
		//if (a1 == 0 && a2 == 0 && a3 == 0)
			return false;

		T t[4];
		int nsol = SolveCubic(a3, a2, a1, a0, t);
		//t[nsol] = 1; // also check at end of timestep

		bool t_flag = false;
		for (int i = 0; i < nsol; i++) {
			if (t[i] < 0 || t[i] > 1)
				continue;

			Vector<T, 3> tx0 = xvpos(x0, v0, t[i]);
			Vector<T, 3> tx1 = xvpos(x1 + x0, v1 + v0, t[i]);
			Vector<T, 3> tx2 = xvpos(x2 + x0, v2 + v0, t[i]);
			Vector<T, 3> tx3 = xvpos(x3 + x0, v3 + v0, t[i]);

			Vector<T, 3> n;
			T w[4];
			T d;
			bool inside;

			if (isVF == 0) {
				d = SignedDistanceVF(tx0, tx1, tx2, tx3, &n, w);
				inside = (fmin(-w[1], fmin(-w[2], -w[3])) >= -1e-6);
			}
			else {// Impact::EE
				d = SignedDistanceEE(tx0, tx1, tx2, tx3, &n, w);
				inside = (fmin(fmin(w[0], w[1]), fmin(-w[2], -w[3])) >= -1e-6);
			}

			if (fabs(d) < 1e-6 && inside)
			{
// 				time = t[i];
// 				return true;
				time = t[i] < time ? t[i] : time;
				t_flag = true;
			}
		}
		return t_flag;
	}

	template<typename T>
	DYN_FUNC bool TightCCD<T>::VertexFaceCCD(
		const Vector<T, 3>& p0, const Vector<T, 3>& a0, const Vector<T, 3>& b0, const Vector<T, 3>& c0,
		const Vector<T, 3>& p1, const Vector<T, 3>& a1, const Vector<T, 3>& b1, const Vector<T, 3>& c1,
		T& time)
	{
		Vector<T, 3> x0 = p0;
		Vector<T, 3> x1 = a0 - p0;
		Vector<T, 3> x2 = b0 - p0;
		Vector<T, 3> x3 = c0 - p0;
		Vector<T, 3> v0 = p1 - p0;
		Vector<T, 3> v1 = a1 - a0 - v0;
		Vector<T, 3> v2 = b1 - b0 - v0;
		Vector<T, 3> v3 = c1 - c0 - v0;

		return CollisionTest(x0, x1, x2, x3, v0, v1, v2, v3, time, 0);
	}

	template<typename T>
	DYN_FUNC bool TightCCD<T>::EdgeEdgeCCD(
		const Vector<T, 3>& a0, const Vector<T, 3>& b0, const Vector<T, 3>& c0, const Vector<T, 3>& d0,
		const Vector<T, 3>& a1, const Vector<T, 3>& b1, const Vector<T, 3>& c1, const Vector<T, 3>& d1,
		T& time)
	{
		Vector<T, 3> p0 = a0;
		Vector<T, 3> p1 = b0 - a0;
		Vector<T, 3> p2 = c0 - a0;
		Vector<T, 3> p3 = d0 - a0;
		Vector<T, 3> v0 = a1 - a0;
		Vector<T, 3> v1 = b1 - b0 - v0;
		Vector<T, 3> v2 = c1 - c0 - v0;
		Vector<T, 3> v3 = d1 - d0 - v0;

		return CollisionTest(a0, p1, p2, p3, v0, v1, v2, v3, time, 1);
	}

	template<typename T>
	DYN_FUNC bool TightCCD<T>::TriangleCCD(TTriangle3D<Real>& s0, TTriangle3D<Real>& s1, TTriangle3D<Real>& t0, TTriangle3D<Real>& t1, Real& toi)
	{
		Real l0 = s0.maximumEdgeLength();
		Real l1 = s1.maximumEdgeLength();
		Real l2 = t0.maximumEdgeLength();
		Real l3 = t1.maximumEdgeLength();

		Real lmax = maximum(maximum(l0, l1), maximum(l2, l3));
		if (lmax < REAL_EPSILON)
			return false;

		Real invL = 1 / lmax;

		Vector<Real, 3> p[3];
		p[0] = invL * s0.v[0];
		p[1] = invL * s0.v[1];
		p[2] = invL * s0.v[2];

		Vector<Real, 3> pp[3];
		pp[0] = invL * s1.v[0];
		pp[1] = invL * s1.v[1];
		pp[2] = invL * s1.v[2];

		Vector<Real, 3> q[3];
		q[0] = invL * t0.v[0];
		q[1] = invL * t0.v[1];
		q[2] = invL * t0.v[2];

		Vector<Real, 3> qq[3];
		qq[0] = invL * t1.v[0];
		qq[1] = invL * t1.v[1];
		qq[2] = invL * t1.v[2];

		///*
		//VF
		bool ret = false;
		for (int st = 0; st < 3; st++)
		{
			Real t = Real(1);
			bool collided = VertexFaceCCD(
				p[st], q[0], q[1], q[2],
				pp[st], qq[0], qq[1], qq[2],
				t);

			toi = collided ? minimum(t, toi) : toi;
			ret |= collided;
		}

		//VF
		for (int st = 0; st < 3; st++)
		{
			Real t = Real(1);
			bool collided = VertexFaceCCD(q[st], p[0], p[1], p[2],
				qq[st], pp[0], pp[1], pp[2],
				t);
			toi = collided ? minimum(t, toi) : toi;
			ret |= collided;
		}

		//EE
		for (int st = 0; st < 3; st++)
		{
			int ind0 = st;
			int ind1 = (st + 1) % 3;
			for (int ss = 0; ss < 3; ss++)
			{
				int ind2 = ss;
				int ind3 = (ss + 1) % 3;

				Real t = Real(1);
				bool collided = EdgeEdgeCCD(
					p[ind0], p[ind1], q[ind2], q[ind3],
					pp[ind0], pp[ind1], qq[ind2], qq[ind3],
					t);

				toi = collided ? minimum(t, toi) : toi;
				ret |= collided;
			}
		}

		return ret;
	}
}