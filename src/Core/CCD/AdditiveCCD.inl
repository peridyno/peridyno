#include "Vector.h"

namespace dyno
{
	#define REAL_infinity 1.0e30
	#define	REAL_EQUAL(a,b)  (((a < b + EPSILON) && (a > b - EPSILON)) ? true : false)
	#define REAL_GREAT(a,b) ((a > EPSILON + b)? true: false) 
	#define REAL_LESS(a,b) ((a + EPSILON < b)? true: false) 
	#define MAX_ITE (500) 

	

	template<typename T>
	DYN_FUNC T getPoint2SegmentDistance( const Vector<T,3> &p, const Vector<T,3>& v0, const Vector<T,3>& v1){
		T d0 = (p - v0).norm();
		T d1 = (p - v1).norm();
		T dv = min(d0, d1);
		if ((v1 - v0).norm() < 1e-7) return dv; //v0 = v1

		Vector<T, 3> dir = (v0 - v1)/(v0-v1).norm();
		if ((p-v0).dot(dir) * (p-v1).dot(dir) <= 0.0) { //project in line
			Vector<T, 3> prox = (v0 - p) - (v0 - p).dot(dir) * dir;
					T dp = prox.norm();
					return dp;
		}
		else
			return dv;
		
	}

	template<typename T>
	DYN_FUNC bool inTri(const Vector<T, 3>& p, 
		const Vector<T, 3>& v0, const Vector<T, 3>& v1, const Vector<T, 3>& v2) {
		auto n = (v1 - v0).cross(v2 - v0);
		if (n.norm() < 1e-6)// in line 
			return false;
		n.normalize();
		// On this board, assuming that p is in plane of triangle, so one have to project the point in plane first.
		auto d0 = (p - v0).cross(v1 - v0);
		auto d1 = (p - v1).cross(v2 - v1);
		auto d2 = (p - v2).cross(v0 - v2);
		return (REAL_LESS(d0.dot(n),0.0) && REAL_LESS(d1.dot(n),0.0) && REAL_LESS(d2.dot(n),0.0)) || 
			(REAL_GREAT(d0.dot(n), 0.0) && REAL_GREAT(d1.dot(n), 0.0) && REAL_GREAT(d2.dot(n), 0.0));
	}

	template<typename T>
	DYN_FUNC T AdditiveCCD<T>::DistanceVF(
		const Vector<T, 3>& x,
		const Vector<T, 3>& y0,
		const Vector<T, 3>& y1,
		const Vector<T, 3>& y2)
	{
		Vector<T, 3> y01 = y1 - y0;
		Vector<T, 3> y12 = y2 - y1;
		Vector<T, 3> n = y01.cross(y12); //norm of plane
		T minLine = min(min(getPoint2SegmentDistance(x, y0, y1), getPoint2SegmentDistance(x, y1, y2)),
			getPoint2SegmentDistance(x, y2, y0));
		if (n.norm() < 1e-6) // in line
			return minLine;

		n.normalize();
		//projection point to plane;
		Vector<T, 3> p_ortho = (x - y0).dot(n) * n;
		Vector<T, 3> p_inPlane = x - p_ortho;
		if (inTri(p_inPlane, y0, y1, y2)) {
			return p_ortho.norm();
		}
		return minLine;
	}

	template<typename T>
	DYN_FUNC Vector<T, 3> AdditiveCCD<T>::DistanceVF_v(
		const Vector<T, 3>& x,
		const Vector<T, 3>& y0,
		const Vector<T, 3>& y1,
		const Vector<T, 3>& y2,
		T* para) 
	{
		//triangle base: y0, e0(y0,y1), e1(y0,y2)
		//point base: x0

		Vector<T, 3> d = y0 - x;
		Vector<T, 3> e0 = y1 - y0;
		Vector<T, 3> e1 = y2 - y0;
		T a00 = e0.dot(e0);
		T a01 = e0.dot(e1);
		T a11 = e1.dot(e1);
		T b0 = e0.dot(d);
		T b1 = e1.dot(d);
		T f = d.dot(d);
		T det = max(a00 * a11 - a01 * a01, 0.0);
		T s = a01 * b1 - a11 * b0;
		T t = a01 * b0 - a00 * b1;
		if (s + t <= det) {
			if (REAL_LESS(s, 0.0)) {
				if (REAL_LESS(t, 0.0)) {
					//region 4
					if (REAL_LESS(b0,0.0)){
						t = 0.0;
						if (-b0 >= a00){
							s = 1.0;
						}
						else{
							s = -b0 / a00;
						}
					}
					else{
						s = 0.0;
						if (REAL_GREAT(b1,0.0)||REAL_EQUAL(b1,0.0)){
							t = 0.0;
						}
						else if (REAL_GREAT(-b1,a11)|| REAL_EQUAL(-b1,a11)){
							t = 1.0;
						}
						else{
							t = -b1 / a11;
						}
					}
				}
				else {
					//region 3
					s = 0.0;
					if (REAL_GREAT(b1,0.0)||REAL_EQUAL(b1,0.0)){
						t = 0.0;
					}
					else if (REAL_GREAT(-b1,a11)||REAL_EQUAL(-b1,a11)){
						t = 1.0;
					}
					else{
						t = -b1 / a11;
					}
				}
			}
			else if (REAL_LESS(t, 0.0)) {
				//region 5
				t = 0.0;
				if (REAL_GREAT(b0,0.0)||REAL_EQUAL(b0,0.0)){
					s = 0.0;
				}
				else if (REAL_GREAT(-b0,a00)||REAL_EQUAL(-b0,a00)){
					s = 1.0;
				}
				else{
					s = -b0 / a00;
				}
			}
			else {
				//region 0, minimum at interior point
				s /= det;
				t /= det;
			}
		}
		else {
			T tmp0 = 0.0; T tmp1 = 0.0; T numer = 0.0; T denom = 0.0;
			if (REAL_LESS(s, 0.0)) {
					//region 2
				tmp0 = a01 + b0;
				tmp1 = a11 + b1;
				if (REAL_GREAT(tmp1,tmp0)){
					numer = tmp1 - tmp0;
					denom = a00 - 2.0 * a01 + a11;
					if (REAL_GREAT(numer,denom)||REAL_EQUAL(numer,denom)){
						s = 1.0;
						t = 0.0;
					}
					else{
						s = numer / denom;
						t = 1.0 - s;
					}
				}
				else
				{
					s = 0.0;
					if (REAL_LESS(tmp1,0.0)||REAL_EQUAL(tmp1,0.0)){
						t = 1.0;
					}
					else if (REAL_GREAT(b1,0.0)||REAL_EQUAL(b1,0.0)){
						t = 0.0;
					}
					else{
						t = -b1 / a11;
					}
				}
			}
			else if (REAL_LESS(t, 0.0)) {
					//region 6
				tmp0 = a01 + b1;
				tmp1 = a00 + b0;
				if (REAL_GREAT(tmp1,tmp0)){
					numer = tmp1 - tmp0;
					denom = a00 - 2.0 * a01 + a11;
					if (REAL_GREAT(numer,denom)||REAL_EQUAL(numer,denom)){
						t = 1.0;
						s = 0.0;
					}
					else
					{
						t = numer / denom;
						s = 1.0 - t;
					}
				}
				else{
					t = 1.0;
					if (REAL_LESS(tmp1,0.0)||REAL_EQUAL(tmp1,0.0)){
						s = 1.0;
					}
					else if (REAL_GREAT(b0,0.0)||REAL_EQUAL(b0,0.0)){
						s = 0.0;
					}
					else{
						s = -b0 / a00;
					}
				}
			}
			else {
				//region 1
				numer = a11 + b1 - a01 - b0;
				if (REAL_LESS(numer,0.0)||REAL_EQUAL(numer,0.0)){
					s = 0.0;
					t = 1.0;
				}
				else{
					denom = a00 - 2.0 * a01 + a11;
					if (REAL_GREAT(numer,denom)||REAL_EQUAL(numer,denom)){
						s = 1.0;
						t = 0.0;
					}
					else{
						s = numer / denom;
						t = 1.0 - s;
					}
				}
			}
		}

		Vector<T, 3> v = x - (y0 + s * e0 + t * e1);
		if (para != nullptr) {
			para[0] = 1.0;
			para[1] = 1.0-s-t;
			para[2] = s;
			para[3] = t;
		}
		
		return v;
	}



	template<typename T>
	DYN_FUNC Vector<T, 3> AdditiveCCD<T>::DistanceEE(
		const Vector<T, 3>& x0, const Vector<T, 3>& x1,
		const Vector<T, 3>& y0, const Vector<T, 3>& y1,
		T* para)
	{
		//segment s:(x0, x1); base: x0, dir: x1-x0
		//segment t:(y0, y1); base: y0, dir: y1-y0

		Vector<T, 3> P1mP0 = x1 - x0;
		Vector<T, 3> Q1mQ0 = y1 - y0;
		Vector<T, 3> P0mQ0  = x0 - y0;
		T a = P1mP0.dot(P1mP0);
		T b = P1mP0.dot(Q1mQ0);
		T c = Q1mQ0.dot(Q1mQ0);
		T d = P1mP0.dot(P0mQ0);
		T e = Q1mQ0.dot(P0mQ0);
		T det = a * c - b * b;
		T s, t, nd, bmd, bte, ctd, bpe, ate, btd;

		if (REAL_GREAT(det, 0.0)) {
			bte = b * e;
			ctd = c * d;
			if (REAL_LESS(bte, ctd) || REAL_EQUAL(bte, ctd)) { //s<=0
				s = 0.0;
				if (REAL_LESS(e, 0.0) || REAL_EQUAL(e, 0.0)) {//t<=0
					//reigen 6
					t = 0.0;
					nd = -d;
					if (REAL_GREAT(nd, a) || REAL_EQUAL(nd, a)) {
						s = 1.0;
					}
					else if (REAL_GREAT(nd, 0.0)) {
						s = nd / a;
					}
				}
				else if (REAL_LESS(e, c))//0<t<1
				{
					//reigon 5
					t = e / c;
				}
				else {//t >=1
					//reigon 4
					t = 1.0;
					bmd = b - d;
					if (REAL_GREAT(bmd, a) || REAL_EQUAL(bmd, a)) {
						s = 1.0;
					}
					else if (REAL_GREAT(bmd, 0.0)) {
						s = bmd / a;
					}
				}
			}
			else { //s>0
				s = bte - ctd;
				if (REAL_GREAT(s, det) || REAL_EQUAL(s, det)) { //s>=1
					s = 1;
					bpe = b + e;
					if (REAL_LESS(bpe, 0.0)||REAL_EQUAL(bpe,0.0)) { //t<=0
						//reigon 8
						t = 0.0;
						nd = -d;
						if (REAL_LESS(nd, 0.0) || REAL_EQUAL(nd, 0.0)) {
							s = 0.0;
						}
						else if (REAL_LESS(nd, a)) {
							s = nd / a;
						}
					}
					else if (REAL_LESS(bpe, c)) {//0<t<1
						//reigon 1
						t = bpe / c;
					}
					else { //t>1
						//reigon 2
						t = 1.0;
						bmd = b - d;
						if (REAL_LESS(bmd, 0.0) || REAL_EQUAL(bmd, 0.0)) {
							s = 0.0;
						}
						else if (REAL_LESS(bmd, a)) {
							s = bmd / a;
						}

					}
				}
				else { //0<s<1
					ate = a * e;
					btd = b * d;
					if (REAL_LESS(ate, btd) || REAL_EQUAL(ate, btd)) { //t<0
						//reigon 7
						t = 0.0;
						nd = -d;
						if (REAL_LESS(nd, 0.0) || REAL_EQUAL(0.0)) {
							s = 0.0;
						}
						else if(REAL_GREAT(nd,a)||REAL_EQUAL(nd,a)){
							s = 1.0;
						}
						else {
							s = nd / a;
						}
					}
					else {//t >0
						t = ate - btd;
						if (REAL_GREAT(t, det) || REAL_EQUAL(t, det)) {
							//region 3
							t = 1.0;
							bmd = b - d;
							if (REAL_LESS(bmd, 0.0) || REAL_EQUAL(bmd, 0.0)) {
								s = 0.0;
							}
							else if (REAL_GREAT(bmd, a) || REAL_EQUAL(bmd, a)) {
								s = 1.0;
							}else{
								s = bmd / a;
							}
						}
						else { //0<t<1
							//reigon 0
							s /= det;
							t /= det;
						}
					}
				}
			}

		}
		else {
		//parallel
		
			if (REAL_LESS(e,0.0)||REAL_EQUAL(e,0.0))  // t <= 0
			{
				// Now solve a*s - b*t + d = 0 for t = 0 (s = -d/a).
				t = 0.0;
				nd = -d;
				if (REAL_LESS(nd, 0.0) || REAL_EQUAL(nd, 0.0))  // s <= 0
				{
					// region 6
					s = 0.0;
				}
				else if (REAL_GREAT(nd,a)||REAL_EQUAL(nd,a))  // s >= 1
				{
					// region 8
					s = 1.0;
				}
				else  // 0 < s < 1
				{
					// region 7
					s = nd / a;
				}
			}
			else if (REAL_GREAT(e,c)||REAL_EQUAL(e,c))  // t >= 1
			{
				// Now solve a*s - b*t + d = 0 for t = 1 (s = (b-d)/a).
				t = 1.0;
				bmd = b - d;
				if (REAL_LESS(bmd,0.0)||REAL_EQUAL(bmd,0.0))  // s <= 0
				{
					// region 4
					s = 0.0;
				}
				else if (REAL_GREAT(bmd,a)||REAL_EQUAL(bmd,a))  // s >= 1
				{
					// region 2
					s = 1.0;
				}
				else  // 0 < s < 1
				{
					// region 3
					s = bmd / a;
				}
			}
			else  // 0 < t < 1
			{
				// The point (0,e/c) is on the line and domain, so we have
				// one point at which R is a minimum.
				s = 0.0;
				t = e / c;
			}
		}

		Vector<T, 3> v = P0mQ0 + s * P1mP0 - (t*Q1mQ0);
		if (para!=nullptr) {
			para[0] = 1.0 - s;;
			para[1] = s;
			para[2] = 1.0-t;
			para[3] = t ;
		}
		return v;
		
	
	}


	//edge(x0,x1), edge(x2,x3)
	template<typename T>
	DYN_FUNC T AdditiveCCD<T>::SquareDistanceEE(
		const Vector<T, 3>& x0, const Vector<T, 3>& x1,
		const Vector<T, 3>& x2, const Vector<T, 3>& x3) {
		
		Vector<T, 3> signedD = DistanceEE(x0, x1, x2, x3, nullptr);
		T  d = signedD.normSquared();
		return d;
	}
	
	//tri(x0,x1,x2), vex(x3)
	template<typename T>
	DYN_FUNC T AdditiveCCD<T>::SquareDistanceVF(const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2,
		const Vector<T, 3>& x3) {
		
		 Vector<T, 3> signedD = DistanceVF_v(x3, x0, x1, x2, nullptr);
		  T d = signedD.normSquared();
		
		// T d = DistanceVF(x3, x0, x1, x2);
		
		return d;
	}
	
	template<typename T>
	DYN_FUNC bool AdditiveCCD<T>::VertexFaceCCD(
		const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2, const Vector<T, 3>& x3,
		const Vector<T, 3>& y0, const Vector<T, 3>& y1, const Vector<T, 3>& y2, const Vector<T, 3>& y3,
		T& time, T invL) {
		//denote p as the transform vector from t0 to t1
		Vector<T, 3> p[4];
		p[0] = y0 - x0;
		p[1] = y1 - x1;
		p[2] = y2 - x2;
		p[3] = y3 - x3;

		Vector<T, 3> x[4];
		x[0] = x0, x[1] = x1; x[2] = x2; x[3] = x3;
		
		Vector<T, 3> p_bar = (p[0] + p[1] + p[2] + p[3]) / 4.0;
		for (int i = 0; i < 4; ++i) {
			p[i] -= p_bar;
		}
		T lp = max(max(p[0].norm(), p[1].norm()), p[2].norm()) + p[3].norm();
		if (lp == 0.0) return false;
		T dsqr = SquareDistanceVF(x[0], x[1], x[2], x[3]);
		//printf("dsqrVF:%f\n", sqrt(dsqr));

		T g = this->s * (dsqr - pow(this->xi * invL, 2)) / (sqrt(dsqr) + this->xi * invL);
		time = 0.0;
		T tL = (1 - this->s) * (dsqr - pow(this->xi * invL, 2)) / ((sqrt(dsqr) + this->xi* invL) * lp);
		
		if (tL<0.0){
			time = 0.0;
			return true;
		}if (tL >= 1.0)
			return false;
			
		int ite = 0;
		while (ite<MAX_ITE) {
			tL = (1 - this->s) * (dsqr - pow(this->xi * invL, 2)) / ((sqrt(dsqr) + this->xi * invL) * lp);
			for (int i = 0; i < 4; ++i)
				x[i] += tL * p[i];
			
			dsqr = SquareDistanceVF(x[0], x[1], x[2], x[3]);
			if (time > 0.0 && ((dsqr - pow(this->xi * invL, 2)) / (sqrt(dsqr) + this->xi * invL) <= g))
				break;
			time += tL;
			if (REAL_GREAT(time, this->tc))
				return false;
			++ite;
			/*if (tL <= 0.0)
				printf("VF, int %d, dsqr-xi^2: %f, lp: %f, tL: %f, time: %f\n", ite, dsqr - pow(this->xi, 2), lp, tL,time);
				*/
		}
		
		//printf("max VF ite:%d\n", ite);
		return true;

	}

	template<typename T>
	DYN_FUNC bool AdditiveCCD<T>::EdgeEdgeCCD(
		const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2, const Vector<T, 3>& x3,
		const Vector<T, 3>& y0, const Vector<T, 3>& y1, const Vector<T, 3>& y2, const Vector<T, 3>& y3,
		T& time, T invL)
	{
		//denote p as the transform vector from t0 to t1
		Vector<T, 3> p[4];
		p[0]=  y0 - x0;
		p[1] = y1 - x1;
		p[2] = y2 - x2;
		p[3] = y3 - x3;
		Vector<T, 3> x[4] = { x0,x1,x2,x3 };
		auto mLength = [](Vector<T, 3>p1, Vector<T, 3> p2, Vector<T, 3>q1, Vector<T, 3> q2) {
			T L1 = pow((p1 - q1).norm(),2);
			T L2 = pow((p1 - q2).norm(),2);
			T L3 = pow((p2 - q1).norm(), 2);
			T L4 = pow((p2 - q2).norm(), 2);
			return minimum(minimum(L1, L2),minimum(L3, L4));
		};
		
		Vector<T, 3> p_bar = (p[0] + p[1] + p[2] + p[3]) / 4.0;
		for (int i = 0; i < 4; ++i) {
			p[i] -= p_bar;
		}

		T lp =	max(p[0].norm(), p[1].norm()) +
				max(p[2].norm(), p[3].norm());

		if (lp ==0.0) return false;

		T dsqr = SquareDistanceEE(x[0], x[1], x[2], x[3]);
		
		/*
		if (dsqr-pow(this->xi,2) <= 0.0)
			dsqr = mLength(x[0], x[1], x[2], x[3]);
			*/
		//printf("dsqrEE:%f\n",sqrt(dsqr));

		T g = this->s * (dsqr - pow(this->xi * invL, 2)) / (sqrt(dsqr) + this->xi * invL);
		time = 0.0;
		T tL = (1 - this->s) * (dsqr - pow(this->xi * invL, 2)) / ((sqrt(dsqr) + this->xi * invL) * lp);
		if (tL < 0.0) {
			time = 0.0;
			return true;
		}if (tL >= 1.0)
			return false;

		int ite = 0;
		while (ite<MAX_ITE) {
			tL = (1 - this->s) * (dsqr - pow(this->xi * invL, 2)) / ((sqrt(dsqr) + this->xi * invL) * lp);

			for (int i = 0; i < 4; ++i) {
				x[i] += tL * p[i];
			}
			
			dsqr = SquareDistanceEE(x[0], x[1], x[2], x[3]);
			/*
			if (dsqr - pow(this->xi, 2) <= 0.0)
				dsqr = mLength(x[0], x[1], x[2], x[3]);
			*/
			auto r = (dsqr - pow(this->xi * invL, 2)) / (sqrt(dsqr) + this->xi * invL);
			if (time >0.0 && r<g)
				break;
			time += tL;
			if (REAL_GREAT(time, this->tc))
				return false;

			++ite;
			
		}
		
		return true;
	}

	template<typename T>
	DYN_FUNC bool AdditiveCCD<T>::TriangleCCD(TTriangle3D<Real>& s0, TTriangle3D<Real>& s1, TTriangle3D<Real>& t0, TTriangle3D<Real>& t1, Real& toi)
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
			bool collided = this->VertexFaceCCD(
				q[0], q[1], q[2],p[st],
				qq[0], qq[1], qq[2],pp[st],
				t, invL);


			toi = collided ? minimum(t, toi) : toi;
			ret |= collided;
		}

		//VF
		for (int st = 0; st < 3; st++)
		{
			Real t = Real(1);
			bool collided = this->VertexFaceCCD(
				p[0], p[1], p[2],q[st],
				pp[0], pp[1], pp[2],qq[st],
				t, invL);
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
				bool collided = this->EdgeEdgeCCD(
					p[ind0], p[ind1], q[ind2], q[ind3],
					pp[ind0], pp[ind1], qq[ind2], qq[ind3],
					t, invL);
			
				toi = collided ? minimum(t, toi) : toi;
				ret |= collided;
			}
		}
		toi = max(toi, 0.0);
		toi = min(toi, 1.0);
		return ret;
	}


	template<typename T>
	DYN_FUNC void AdditiveCCD<T>::projectClosePoint(
		const TTriangle3D<Real>& s, const TTriangle3D<Real>& t,
		Vector<T, 3>& first, Vector<T, 3>& second) {

		Real l0 = s.maximumEdgeLength();
		Real l1 = t.maximumEdgeLength();
		

		Real lmax = maximum(l0, l1);
		if (lmax < REAL_EPSILON) { //triangles are  as small as points.
			first[0] = 1.0, first[1] = 0.0, first[2] = 0.0,
			second[0] = 1.0, second[1] = 0.0, second[2] = 0.0;
			return;
		}

		Real invL = 1 / lmax;

		Vector<Real, 3> p[3];
		p[0] = invL * s.v[0];
		p[1] = invL * s.v[1];
		p[2] = invL * s.v[2];

		Vector<Real, 3> q[3];
		q[0] = invL * t.v[0];
		q[1] = invL * t.v[1];
		q[2] = invL * t.v[2];
                                                                     
		//VF
		Real D = REAL_infinity;

		for (int st = 0; st < 3; st++)
		{
			T para[4];
			auto disVector = this->DistanceVF_v(
				p[st], q[0], q[1], q[2], para);
			auto dis = disVector.norm();
			if (dis < D) {
				D = dis;
				first[st] = para[0];
				first[(st + 1) % 3] = 0.0;
				first[(st + 2) % 3] = 0.0;
				second[0] = para[1];
				second[1] = para[2];
				second[2] = para[3];
			}
		}
		
		//VF
		for (int st = 0; st < 3; st++)
		{
			T para[4];
			auto disVector = this->DistanceVF_v(
				q[st], p[0], p[1], p[2], para);
			auto dis = disVector.norm();
			if (dis < D) {
				D = dis;
				second[st] = para[0];
				second[(st + 1) % 3] = 0.0;
				second[(st + 2) % 3] = 0.0;
				first[0] = para[1];
				first[1] = para[2];
				first[2] = para[3];
			}
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

				T para[4];
				auto disVector = this->DistanceEE(
					p[ind0], p[ind1], q[ind2], q[ind3], para);
				auto dis = disVector.norm();
				if (dis <= (D+EPSILON)) {
					D = dis;
					first[ind0] = para[0];
					first[ind1] = para[1];
					first[(ind1 + 1) % 3] = 0.0;
					second[ind2] = para[2];
					second[ind3] = para[3];
					second[(ind3 + 1) % 3] = 0.0;
				}
			}
		}

	 }//end function




}//end inl