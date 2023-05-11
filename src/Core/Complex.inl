#include "Math/SimpleMath.h"
#include <glm/glm.hpp>
#include "Primitive/Primitive3D.h"

namespace dyno
{
	template <typename Real>
	DYN_FUNC Complex<Real>::Complex() :
		m_real(0.0), m_imag(0.0)
	{
	}

	template <typename Real>
	DYN_FUNC Complex<Real>::Complex(Real real, Real imag) :
		m_real(real), m_imag(imag)
	{
	}

	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator+ (const Complex<Real> &other) const {
		Complex<Real> sum;
		sum.m_real = m_real + other.m_real;
		sum.m_imag = m_imag + other.m_imag;
		return sum;
	}

	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator- (const Complex<Real> &other) const {
		Complex dif;
		dif.m_real = m_real - other.m_real;
		dif.m_imag = m_imag - other.m_imag;
		return dif;
	}

	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator* (const Complex<Real> &other) const
	{
		Complex mul;
		mul.m_real = (m_real * other.m_real - m_imag * other.m_imag);
		mul.m_imag = (m_real * other.m_imag + m_imag * other.m_real);
		return mul;
	}

	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator/(const Complex<Real> &other) const
	{
		Complex div;
		double l = (other.m_real*other.m_real + other.m_imag*other.m_imag);

		

		double c = (m_real*other.m_real + m_imag * other.m_imag) / l;
		double d = (m_imag*other.m_real - m_real * other.m_imag) / l;
		div.m_real = c;
		div.m_imag = d;
		return div;
	}

	template <typename Real>
	DYN_FUNC Complex<Real> &Complex<Real>::operator+=(const Complex<Real> &other)
	{
		m_real += other.realPart();
		m_imag += other.imagPart();
		return *this;
	}

	template <typename Real>
	DYN_FUNC Complex<Real> &Complex<Real>::operator-=(const Complex<Real> &other)
	{
		m_real -= other.realPart();
		m_imag -= other.imagPart();
		return *this;
	}

	template <typename Real>
	DYN_FUNC Complex<Real> &Complex<Real>::operator*=(const Complex &other)
	{
		double j = m_real * other.m_real + m_imag * other.m_imag;
		double k = m_real * other.m_imag + m_imag * other.m_real;
		m_real = j;
		m_imag = k;
		return *this;
	}

	template <typename Real>
	DYN_FUNC Complex<Real> &Complex<Real>::operator/=(const Complex<Real> &other)
	{
		double a = (m_real*other.m_real + m_imag * other.m_imag) /
			(other.m_real*other.m_real + other.m_imag*other.m_imag);
		double b = (m_imag*other.m_real - m_real * other.m_imag) / (other.m_real*other.m_real + other.m_imag*other.m_imag);
		m_real = a;
		m_imag = b;
		return *this;
	}


	template <typename Real>
	DYN_FUNC Complex<Real>& Complex<Real>::operator=(const Complex<Real> &other)
	{
		m_real = other.m_real;
		m_imag = other.m_imag;

		return *this;
	}


	template <typename Real>
	DYN_FUNC bool Complex<Real>::operator==(const Complex<Real> &other) const
	{
		if (glm::abs(m_real - other.m_real) < dyno::REAL_EPSILON && glm::abs(m_imag - other.m_imag) < dyno::REAL_EPSILON) {
			return true;
		}
		else {
			return false;
		}
		return false;
	}

	template <typename Real>
	DYN_FUNC bool Complex<Real>::operator!=(const Complex<Real> &other) const
	{
		return !(*this == other);
	}


	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator+(const Real& real) const
	{
		Complex<Real> addition;
		addition.m_real = m_real + real;
		addition.m_imag = m_imag;
		return addition;
	}

	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator-(const Real& real) const
	{
		Complex<Real> sub;
		sub.m_real = m_real - real;
		sub.m_imag = m_imag;
		return sub;
	}

	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator*(const Real& real) const
	{
		Complex<Real> mul;
		mul.m_real = m_real * real;
		mul.m_imag = m_imag * real;
		return mul;
	}


	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator/(const Real& real) const
	{
		Complex<Real> div;
		div.m_real = m_real / real;
		div.m_imag = m_imag / real;
		return div;
	}

	template <typename Real>
	DYN_FUNC Complex<Real>& Complex<Real>::operator+=(const Real& real)
	{
		m_real += real;
		return *this;
	}


	template <typename Real>
	DYN_FUNC Complex<Real>& Complex<Real>::operator-=(const Real& real)
	{
		m_real -= real;
		return *this;
	}

	template <typename Real>
	DYN_FUNC Complex<Real>& Complex<Real>::operator*=(const Real& real)
	{
		m_real *= real;
		m_imag *= real;
		return *this;
	}


	template <typename Real>
	DYN_FUNC Complex<Real>& Complex<Real>::operator/=(const Real& real)
	{
		m_real /= real;
		m_imag /= real;
		return *this;
	}


	template <typename Real>
	DYN_FUNC const Complex<Real> Complex<Real>::operator-(void) const
	{
		Complex<Real> neg;
		neg.m_real = -m_real;
		neg.m_imag = -m_imag;
		return neg;
	}


	template <typename Real>
	DYN_FUNC Complex<Real> Complex<Real>::conjugate() const
	{
		Complex<Real> res(m_real, -m_imag);
		return res;
	}


	template <typename Real>
	DYN_FUNC Real Complex<Real>::norm() const
	{
		return glm::sqrt(m_real*m_real + m_imag * m_imag);
	}


	template <typename Real>
	DYN_FUNC Real Complex<Real>::normSquared() const
	{
		return m_real * m_real + m_imag * m_imag;
	}

	//Relax the threshold by 10 times 
	template <typename Real>
	DYN_FUNC bool Complex<Real>::isReal() const
	{
		return glm::abs(m_imag) < Real(10)*REAL_EPSILON;
	}

	template <typename S, typename T>
	DYN_FUNC const Complex<T> operator +(S scale, const Complex<T> &complex)
	{
		return complex + (T)scale;
	}

	template <typename S, typename T>
	DYN_FUNC const Complex<T> operator -(S scale, const Complex<T> &complex)
	{
		return Complex<T>((T)scale, T(0)) - complex;
	}

	template <typename S, typename T>
	DYN_FUNC const Complex<T> operator *(S scale, const Complex<T> &complex)
	{
		return complex * (T)scale;
	}

	template <typename S, typename T>
	DYN_FUNC const Complex<T> operator /(S scale, const Complex<T> &complex)
	{
		return Complex<T>((T)scale, T(0)) / complex;
	}


	template<class Real>
	DYN_FUNC Complex<Real> sinh(const Complex<Real>& __x)
	{
		if (isinf(__x.realPart()) && !isfinite(__x.imagPart()))
			return Complex<Real>(__x.realPart(), Real(NAN));
		if (__x.realPart() == 0 && !isfinite(__x.imagPart()))
			return Complex<Real>(__x.realPart(), Real(NAN));
		if (__x.imagPart() == 0 && !isfinite(__x.realPart()))
			return __x;
		return Complex<Real>(sinh(__x.realPart()) * cos(__x.imagPart()), cosh(__x.realPart()) * sin(__x.imagPart()));
	}


	template<typename Real>
	inline DYN_FUNC Real arg(const Complex<Real>& __c)
	{
		return atan2(__c.imagPart(), __c.realPart());
	}

	template<typename Real>
	inline DYN_FUNC Complex<Real> polar(const Real& __rho, const Real& __theta)
	{
		if (isnan(__rho) || signbit(__rho))
			return Complex<Real>(Real(NAN), Real(NAN));
		if (isnan(__theta))
		{
			if (isinf(__rho))
				return Complex<Real>(__rho, __theta);
			return Complex<Real>(__theta, __theta);
		}
		if (isinf(__theta))
		{
			if (isinf(__rho))
				return Complex<Real>(__rho, Real(NAN));
			return Complex<Real>(Real(NAN), Real(NAN));
		}
		Real __x = __rho * glm::cos(__theta);
		if (isnan(__x))
			__x = 0;
		Real __y = __rho * glm::sin(__theta);
		if (isnan(__y))
			__y = 0;
		return Complex<Real>(__x, __y);
	}

	template<typename Real>
	inline DYN_FUNC Complex<Real> 	log(const Complex<Real>& __x)
	{
		return Complex<Real>(glm::log(__x.norm()), arg(__x));
	}

	// log10

	template<typename Real>
	inline DYN_FUNC Complex<Real> log10(const Complex<Real>& __x)
	{
		return log(__x) / log(Real(10));
	}

	// sqrt

	template<typename Real>
	inline DYN_FUNC Complex<Real> sqrt(const Complex<Real>& __x)
	{
		if (isinf(__x.imagPart()))
			return Complex<Real>(Real(INFINITY), __x.imagPart());
		if (isinf(__x.realPart()))
		{
			if (__x.realPart() > Real(0))
				return Complex<Real>(__x.realPart(), isnan(__x.imagPart()) ? __x.imagPart() : copysign(Real(0), __x.imagPart()));
			return Complex<Real>(isnan(__x.imagPart()) ? __x.imagPart() : Real(0), copysign(__x.realPart(), __x.imagPart()));
		}

		return polar(glm::sqrt(__x.norm()), arg(__x) / Real(2));
	}

	// exp

	template<class Real>
	DYN_FUNC Complex<Real>	exp(const Complex<Real>& __x)
	{
		Real __i = __x.imagPart();
		if (isinf(__x.realPart()))
		{
			if (__x.realPart() < Real(0))
			{
				if (!isfinite(__i))
					__i = Real(1);
			}
			else if (__i == 0 || !isfinite(__i))
			{
				if (isinf(__i))
					__i = Real(NAN);
				return Complex<Real>(__x.realPart(), __i);
			}
		}
		else if (isnan(__x.realPart()) && __x.imagPart() == 0)
			return __x;
		Real __e = glm::exp(__x.realPart());
		return Complex<Real>(__e * glm::cos(__i), __e * glm::sin(__i));
	}

	// pow

	template<class Real>
	inline DYN_FUNC Complex<Real>	pow(const Complex<Real>& __x, const Complex<Real>& __y)
	{
		return exp(__y * log(__x));
	}

	template<class Real>
	inline DYN_FUNC Complex<Real>	pow(const Complex<Real>& __x, const Real& __y)
	{
		return pow(__x, Complex<Real>(__y));
	}

	template<class Real>
	inline DYN_FUNC Complex<Real>	pow(const Real& __x, const Complex<Real>& __y)
	{
		return pow(Complex<Real>(__x), __y);
	}

	// asinh

	template<class Real>
	DYN_FUNC Complex<Real>	asinh(const Complex<Real>& __x)
	{
		const Real __pi(atan2(+0., -0.));
		if (isinf(__x.realPart()))
		{
			if (isnan(__x.imagPart()))
				return __x;
			if (isinf(__x.imagPart()))
				return Complex<Real>(__x.realPart(), copysign(__pi * Real(0.25), __x.imagPart()));
			return Complex<Real>(__x.realPart(), copysign(Real(0), __x.imagPart()));
		}
		if (isnan(__x.realPart()))
		{
			if (isinf(__x.imagPart()))
				return Complex<Real>(__x.imagPart(), __x.realPart());
			if (__x.imagPart() == 0)
				return __x;
			return Complex<Real>(__x.realPart(), __x.realPart());
		}
		if (isinf(__x.imagPart()))
			return Complex<Real>(copysign(__x.imagPart(), __x.realPart()), copysign(__pi / Real(2), __x.imagPart()));
		Complex<Real> __z = log(__x + sqrt(pow(__x, Real(2)) + Real(1)));
		return Complex<Real>(copysign(__z.realPart(), __x.realPart()), copysign(__z.imagPart(), __x.imagPart()));
	}

	template<class Real>
	DYN_FUNC Complex<Real>	cosh(const Complex<Real>& __x)
	{
		if (isinf(__x.realPart()) && !isfinite(__x.imagPart()))
			return Complex<Real>(glm::abs(__x.realPart()), Real(NAN));
		if (__x.realPart() == 0 && !isfinite(__x.imagPart()))
			return Complex<Real>(Real(NAN), __x.realPart());
		if (__x.realPart() == 0 && __x.imagPart() == 0)
			return Complex<Real>(Real(1), __x.imagPart());
		if (__x.imagPart() == 0 && !isfinite(__x.realPart()))
			return Complex<Real>(glm::abs(__x.realPart()), __x.imagPart());
		return Complex<Real>(glm::cosh(__x.realPart()) * glm::cos(__x.imagPart()), glm::sinh(__x.realPart()) * glm::sin(__x.imagPart()));
	}


	// acosh

	template<class Real>
	DYN_FUNC Complex<Real> acosh(const Complex<Real>& __x)
	{
		const Real __pi(atan2(+0., -0.));
		if (isinf(__x.realPart()))
		{
			if (isnan(__x.imagPart()))
				return Complex<Real>(fabs(__x.realPart()), __x.imagPart());
			if (isinf(__x.imagPart()))
				if (__x.realPart() > 0)
					return Complex<Real>(__x.realPart(), copysign(__pi * Real(0.25), __x.imagPart()));
				else
					return Complex<Real>(-__x.realPart(), copysign(__pi * Real(0.75), __x.imagPart()));
			if (__x.realPart() < 0)
				return Complex<Real>(-__x.realPart(), copysign(__pi, __x.imagPart()));
			return Complex<Real>(__x.realPart(), copysign(Real(0), __x.imagPart()));
		}
		if (isnan(__x.realPart()))
		{
			if (isinf(__x.imagPart()))
				return Complex<Real>(fabs(__x.imagPart()), __x.realPart());
			return Complex<Real>(__x.realPart(), __x.realPart());
		}
		if (isinf(__x.imagPart()))
			return Complex<Real>(fabs(__x.imagPart()), copysign(__pi / Real(2), __x.imagPart()));
		Complex<Real> __z = log(__x + sqrt(pow(__x, Real(2)) - Real(1)));
		return Complex<Real>(copysign(__z.realPart(), Real(0)), copysign(__z.imagPart(), __x.imagPart()));
	}

	// atanh

	template<class Real>
	DYN_FUNC Complex<Real> atanh(const Complex<Real>& __x)
	{
		const Real __pi(atan2(+0., -0.));
		if (isinf(__x.imagPart()))
		{
			return Complex<Real>(copysign(Real(0), __x.realPart()), copysign(__pi / Real(2), __x.imagPart()));
		}
		if (isnan(__x.imagPart()))
		{
			if (isinf(__x.realPart()) || __x.realPart() == 0)
				return Complex<Real>(copysign(Real(0), __x.realPart()), __x.imagPart());
			return Complex<Real>(__x.imagPart(), __x.imagPart());
		}
		if (isnan(__x.realPart()))
		{
			return Complex<Real>(__x.realPart(), __x.realPart());
		}
		if (isinf(__x.realPart()))
		{
			return Complex<Real>(copysign(Real(0), __x.realPart()), copysign(__pi / Real(2), __x.imagPart()));
		}
		if (fabs(__x.realPart()) == Real(1) && __x.imagPart() == Real(0))
		{
			return Complex<Real>(copysign(Real(INFINITY), __x.realPart()), copysign(Real(0), __x.imagPart()));
		}
		Complex<Real> __z = log((Real(1) + __x) / (Real(1) - __x)) / Real(2);
		return Complex<Real>(copysign(__z.realPart(), __x.realPart()), copysign(__z.imagPart(), __x.imagPart()));
	}


	// tanh

	template<class Real>
	DYN_FUNC Complex<Real>	tanh(const Complex<Real>& __x)
	{
		if (isinf(__x.realPart()))
		{
			if (!isfinite(__x.imagPart()))
				return Complex<Real>(Real(1), Real(0));
			return Complex<Real>(Real(1), copysign(Real(0), sin(Real(2) * __x.imagPart())));
		}
		if (isnan(__x.realPart()) && __x.imagPart() == 0)
			return __x;
		Real __2r(Real(2) * __x.realPart());
		Real __2i(Real(2) * __x.imagPart());
		Real __d(cosh(__2r) + cos(__2i));
		return  Complex<Real>(sinh(__2r) / __d, sin(__2i) / __d);
	}

	// asin

	template<class Real>
	DYN_FUNC Complex<Real>	asin(const Complex<Real>& __x)
	{
		Complex<Real> __z = asinh(Complex<Real>(-__x.imagPart(), __x.realPart()));
		return Complex<Real>(__z.imagPart(), -__z.realPart());
	}

	// acos

	template<class Real>
	DYN_FUNC Complex<Real>	acos(const Complex<Real>& __x)
	{
		const Real __pi(atan2(+0., -0.));
		if (isinf(__x.realPart()))
		{
			if (isnan(__x.imagPart()))
				return Complex<Real>(__x.imagPart(), __x.realPart());
			if (isinf(__x.imagPart()))
			{
				if (__x.realPart() < Real(0))
					return Complex<Real>(Real(0.75) * __pi, -__x.imagPart());
				return Complex<Real>(Real(0.25) * __pi, -__x.imagPart());
			}
			if (__x.realPart() < Real(0))
				return Complex<Real>(__pi, signbit(__x.imagPart()) ? -__x.realPart() : __x.realPart());
			return Complex<Real>(Real(0), signbit(__x.imagPart()) ? __x.realPart() : -__x.realPart());
		}
		if (isnan(__x.realPart()))
		{
			if (isinf(__x.imagPart()))
				return Complex<Real>(__x.realPart(), -__x.imagPart());
			return Complex<Real>(__x.realPart(), __x.realPart());
		}
		if (isinf(__x.imagPart()))
			return Complex<Real>(__pi / Real(2), -__x.imagPart());
		if (__x.realPart() == 0)
			return Complex<Real>(__pi / Real(2), -__x.imagPart());
		Complex<Real> __z = log(__x + sqrt(pow(__x, Real(2)) - Real(1)));
		if (signbit(__x.imagPart()))
			return Complex<Real>(fabs(__z.imagPart()), fabs(__z.realPart()));
		return Complex<Real>(fabs(__z.imagPart()), -fabs(__z.realPart()));
	}

	// atan

	template<class Real>
	DYN_FUNC Complex<Real> atan(const Complex<Real>& __x)
	{
		Complex<Real> __z = atanh(Complex<Real>(-__x.imagPart(), __x.realPart()));
		return Complex<Real>(__z.imagPart(), -__z.realPart());
	}

	// sin

	template<class Real>
	DYN_FUNC Complex<Real>	sin(const Complex<Real>& __x)
	{
		Complex<Real> __z = sinh(Complex<Real>(-__x.imagPart(), __x.realPart()));
		return Complex<Real>(__z.imagPart(), -__z.realPart());
	}

	// cos

	template<class Real>
	inline DYN_FUNC Complex<Real>	cos(const Complex<Real>& __x)
	{
		return cosh(Complex<Real>(-__x.imagPart(), __x.realPart()));
	}

	// tan

	template<class Real>
	DYN_FUNC Complex<Real>	tan(const Complex<Real>& __x)
	{
		Complex<Real> __z = tanh(Complex<Real>(-__x.imagPart(), __x.realPart()));
		return Complex<Real>(__z.imagPart(), -__z.realPart());
	}


}

