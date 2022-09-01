#pragma once
#include "Platform.h"
#include "DeclareEnum.h"

namespace dyno {

	template<typename Real>
	class Kernel
	{
	public:
		DYN_FUNC Kernel() {};
		DYN_FUNC ~Kernel() {};

		DYN_FUNC inline virtual Real Weight(const Real r, const Real h)
		{
			return Real(0);
		}

		DYN_FUNC inline virtual Real Gradient(const Real r, const Real h)
		{
			return Real(0);
		}

		DYN_FUNC inline virtual Real integral(const Real r, const Real h)
		{
			return Real(0);
		}

	public:
		Real m_scale = Real(1);
	};

	//spiky kernel
	template<typename Real>
	class SpikyKernel : public Kernel<Real>
	{
	public:
		DYN_FUNC SpikyKernel() : Kernel<Real>() {};
		DYN_FUNC ~SpikyKernel() {};

		DYN_FUNC inline Real Weight(const Real r, const Real h) override
		{
// 			const Real q = r / h;
// 			if (q > 1.0f) return 0.0f;
// 			else {
// 				const Real d = Real(1) - q;
// 				const Real hh = h*h;
// 				return 15.0f / ((Real)M_PI * hh * h) * d * d * d * this->m_scale;
// 			}
			return SpikyKernel<Real>::weight(r, h, m_scale);
		}

		DYN_FUNC inline Real Gradient(const Real r, const Real h) override
		{
// 			const Real q = r / h;
// 			if (q > 1.0f) return 0.0;
// 			//else if (r==0.0f) return 0.0f;
// 			else {
// 				const Real d = Real(1) - q;
// 				const Real hh = h*h;
// 				return -45.0f / ((Real)M_PI * hh*h) *d*d * this->m_scale;
// 			}
			return SpikyKernel<Real>::gradient(r, h, m_scale);
		}

		DYN_FUNC inline Real integral(const Real r, const Real h) override
		{
			return SpikyKernel<Real>::integral(r, h, m_scale);
		}

		DYN_FUNC static inline Real weight(const Real r, const Real h, Real scale)
		{
			const Real q = r / h;
			if (q > 1.0f) return 0.0f;
			else {
				const Real d = Real(1) - q;
				const Real hh = h * h;
				return 15.0f / ((Real)M_PI * hh * h) * d * d * d * scale;
			}
		}

		DYN_FUNC static inline Real gradient(const Real r, const Real h, Real scale)
		{
			const Real q = r / h;
			if (q > 1.0f) return 0.0;
			//else if (r==0.0f) return 0.0f;
			else {
				const Real d = Real(1) - q;
				const Real hh = h * h;
				return -45.0f / ((Real)M_PI * hh*h) *d*d * scale;
			}
		}

		DYN_FUNC static inline Real integral(const Real r, const Real h, Real scale)
		{
			const Real q = r / h;
			if (q > 1.0f) return 0.0f;
			else {
				const Real qq = q * q;
				const Real hh = h * h;
				return -15.0f / ((Real)M_PI * hh) * (q - Real(1.5) * qq + q * qq - Real(0.25) * qq * qq - Real(0.25)) * scale;
			}
		}
	};

	template<typename Real>
	class ConstantKernel : public Kernel<Real>
	{
	public:
		DYN_FUNC ConstantKernel() : Kernel<Real>() {};
		DYN_FUNC ~ConstantKernel() {};

		DYN_FUNC inline Real Weight(const Real r, const Real h) override
		{
			return Real(1);
		}

		DYN_FUNC inline Real Gradient(const Real r, const Real h) override
		{
			return Real(0);
		}
	};


	template<typename Real>
	class SmoothKernel : public Kernel<Real>
	{
	public:
		DYN_FUNC SmoothKernel() : Kernel<Real>() {};
		DYN_FUNC ~SmoothKernel() {};

		DYN_FUNC inline Real Weight(const Real r, const Real h) override
		{
// 			const Real q = r / h;
// 			if (q > 1.0f) return 0.0f;
// 			else {
// 				return (1.0f - q*q) * this->m_scale;
// 			}
			return SmoothKernel<Real>::weight(r, h, m_scale);
		}

		DYN_FUNC inline Real Gradient(const Real r, const Real h) override
		{
// 			const Real q = r / h;
// 			if (q > Real(1)) return Real(0);
// 			else {
// 				const Real hh = h*h;
// 				const Real dd = Real(1) - q*q;
// 				const Real alpha = 1.0f;// (Real) 945.0f / (32.0f * (Real)M_PI * hh *h);
// 				return -alpha * dd* this->m_scale;
// 			}
			return SmoothKernel<Real>::gradient(r, h, m_scale);
		}

		DYN_FUNC static inline Real weight(const Real r, const Real h, const Real scale)
		{
			const Real q = r / h;
			if (q > 1.0f) return 0.0f;
			else {
				return scale * (1.0f - q * q);
			}
		}

		DYN_FUNC static inline Real gradient(const Real r, const Real h, const Real scale)
		{
			const Real q = r / h;
			if (q > Real(1)) return Real(0);
			else {
				const Real hh = h * h;
				const Real dd = Real(1) - q * q;
				const Real alpha = 1.0f;// (Real) 945.0f / (32.0f * (Real)M_PI * hh *h);
				return -alpha * dd* scale;
			}
		}

		DYN_FUNC static inline Real integral(const Real r, const Real h, Real scale)
		{
			const Real q = r / h;
			if (q > Real(1)) return Real(0);
			else {
				const Real hh = h * h;
				return 1.0 / (hh * h) * (Real(2) / 3 - q + q * q / Real(3));
			}
		}
	};


	//spiky kernel
	template<typename Real>
	class CorrectedKernel : public Kernel<Real>
	{
	public:
		DYN_FUNC CorrectedKernel() : Kernel<Real>() {};
		DYN_FUNC ~CorrectedKernel() {};

		DYN_FUNC inline Real Weight(const Real r, const Real h) override
		{
			const Real q = r / h;
			SmoothKernel<Real> kernSmooth;
			return q*q*q*kernSmooth.Weight(r, h);
		}

		DYN_FUNC inline Real WeightR(const Real r, const Real h)
		{
			const Real q = r / h;
			SmoothKernel<Real> kernSmooth;
			return q*q*kernSmooth.Weight(r, h)/h;
		}

		DYN_FUNC inline Real WeightRR(const Real r, const Real h)
		{
			const Real q = r / h;
			SmoothKernel<Real> kernSmooth;
			return q*kernSmooth.Weight(r, h) / (h*h);
		}
	};

	//cubic kernel
	template<typename Real>
	class CubicKernel : public Kernel<Real>
	{
	public:
		DYN_FUNC CubicKernel() : Kernel<Real>() {};
		DYN_FUNC ~CubicKernel() {};

		DYN_FUNC inline Real Weight(const Real r, const Real h) override
		{
			const Real hh = h*h;
			const Real q = 2.0f*r / h;

			const Real alpha = 3.0f / (2.0f * (Real)M_PI * hh * h);

			if (q > 2.0f) return 0.0f;
			else if (q >= 1.0f)
			{
				//1/6*(2-q)*(2-q)*(2-q)
				const Real d = 2.0f - q;
				return alpha / 6.0f*d*d*d;
			}
			else
			{
				//(2/3)-q*q+0.5f*q*q*q
				const Real qq = q*q;
				const Real qqq = qq*q;
				return alpha*(2.0f / 3.0f - qq + 0.5f*qqq);
			}
		}

		DYN_FUNC inline Real Gradient(const Real r, const Real h) override
		{
			const Real hh = h*h;
			const Real q = 2.0f*r / h;

			const Real alpha = 3.0f / (2.0f * (Real)M_PI * hh * h);

			if (q > 2.0f) return Real(0);
			else if (q >= 1.0f)
			{
				//-0.5*(2.0-q)*(2.0-q)
				const Real d = 2.0f - q;
				return -0.5f*alpha*d*d;
			}
			else
			{
				//-2q+1.5*q*q
				const Real qq = q*q;
				return alpha*(-2.0f*q + 1.5f*qq);
				//return alpha*(-0.5);
			}
		}
	};

	template<typename Real>
	class QuarticKernel : public Kernel<Real>
	{
	public:
		DYN_FUNC QuarticKernel() : Kernel<Real>() {};
		DYN_FUNC ~QuarticKernel() {};

		DYN_FUNC inline Real Weight(const Real r, const Real h) override
		{
			const Real hh = h*h;
			const Real q = 2.5f*r / h;
			if (q > 2.5) return 0.0f;
			else if (q > 1.5f)
			{
				const Real d = 2.5f - q;
				const Real dd = d*d;
				return 0.0255f*dd*dd / hh;
			}
			else if (q > 0.5f)
			{
				const Real d = 2.5f - q;
				const Real t = 1.5f - q;
				const Real dd = d*d;
				const Real tt = t*t;
				return 0.0255f*(dd*dd - 5.0f*tt*tt) / hh;
			}
			else
			{
				const Real d = 2.5f - q;
				const Real t = 1.5f - q;
				const Real w = 0.5f - q;
				const Real dd = d*d;
				const Real tt = t*t;
				const Real ww = w*w;
				return 0.0255f*(dd*dd - 5.0f*tt*tt + 10.0f*ww*ww) / hh;
			}
		}

		DYN_FUNC inline Real Gradient(const Real r, const Real h) override
		{
			const Real hh = h*h;
			const Real q = 2.5f*r / h;
			if (q > 2.5) return 0.0f;
			else if (q > 1.5f)
			{
				//0.102*(2.5-q)^3
				const Real d = 2.5f - q;
				return -0.102f*d*d*d / hh;
			}
			else if (q > 0.5f)
			{
				const Real d = 2.5f - q;
				const Real t = 1.5f - q;
				return -0.102f*(d*d*d - 5.0f*t*t*t) / hh;
			}
			else
			{
				const Real d = 2.5f - q;
				const Real t = 1.5f - q;
				const Real w = 0.5f - q;
				return -0.102f*(d*d*d - 5.0f*t*t*t + 10.0f*w*w*w) / hh;
			}
		}
	};
}
