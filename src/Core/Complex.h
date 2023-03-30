/**
 * Copyright 2021 Xiaowei He
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
 */
#pragma once
#include "Platform.h"

namespace dyno
{
	template <typename Real>
	class Complex
	{
	public:
		DYN_FUNC Complex();
		DYN_FUNC explicit Complex(Real real, Real imag = Real(0));
		DYN_FUNC Real realPart() const { return m_real; }
		DYN_FUNC Real imagPart() const { return m_imag; }

		DYN_FUNC Real& realPart() { return m_real; }
		DYN_FUNC Real& imagPart() { return m_imag; }

		DYN_FUNC Complex<Real> conjugate() const;
		DYN_FUNC Real norm() const;
		DYN_FUNC Real normSquared() const;

		DYN_FUNC bool isReal() const;

		DYN_FUNC const Complex<Real> operator+ (const Complex<Real> &other) const;
		DYN_FUNC const Complex<Real> operator- (const Complex<Real> &other) const;
		DYN_FUNC const Complex<Real> operator* (const Complex<Real> &other) const;
		DYN_FUNC const Complex<Real> operator/ (const Complex<Real> &other) const;

		DYN_FUNC Complex<Real>& operator+= (const Complex<Real> &other);
		DYN_FUNC Complex<Real>& operator-= (const Complex<Real> &other);
		DYN_FUNC Complex<Real>& operator*= (const Complex<Real> &other);
		DYN_FUNC Complex<Real>& operator/= (const Complex<Real> &other);


		DYN_FUNC Complex<Real>& operator= (const Complex<Real> &other);

		DYN_FUNC bool operator== (const Complex<Real> &other) const;
		DYN_FUNC bool operator!= (const Complex<Real> &other) const;

		DYN_FUNC const Complex<Real> operator+ (const Real& real) const;
		DYN_FUNC const Complex<Real> operator- (const Real& real) const;
		DYN_FUNC const Complex<Real> operator* (const Real& real) const;
		DYN_FUNC const Complex<Real> operator/ (const Real& real) const;

		DYN_FUNC Complex<Real>& operator+= (const Real& real);
		DYN_FUNC Complex<Real>& operator-= (const Real& real);
		DYN_FUNC Complex<Real>& operator*= (const Real& real);
		DYN_FUNC Complex<Real>& operator/= (const Real& real);

		DYN_FUNC const Complex<Real> operator - (void) const;

	protected:
		Real m_real;
		Real m_imag;
	};

 	template<typename Real> inline DYN_FUNC Complex<Real> acos(const Complex<Real>&);
 	template<typename Real> inline DYN_FUNC Complex<Real> asin(const Complex<Real>&);
 	template<typename Real> inline DYN_FUNC Complex<Real> atan(const Complex<Real>&);
 	template<typename Real> inline DYN_FUNC Complex<Real> asinh(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> acosh(const Complex<Real>&);
 	template<typename Real> inline DYN_FUNC Complex<Real> atanh(const Complex<Real>&);
 	template<typename Real> inline DYN_FUNC Complex<Real> cos(const Complex<Real>&);
 	template<typename Real> inline DYN_FUNC Complex<Real> cosh(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> exp(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> log(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> log10(const Complex<Real>&);

	template<typename Real> inline DYN_FUNC Complex<Real> pow(const Complex<Real>&, const Real&);
	template<typename Real> inline DYN_FUNC Complex<Real> pow(const Complex<Real>&, const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> pow(const Real&, const Complex<Real>&);
	// 
	template<typename Real> inline DYN_FUNC Complex<Real> sin(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> sinh(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> sqrt(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> tan(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> tanh(const Complex<Real>&);

	template<typename Real> inline DYN_FUNC Real arg(const Complex<Real>&);
	template<typename Real> inline DYN_FUNC Complex<Real> polar(const Real& __rho, const Real& __theta = Real(0));
}

#include "Complex.inl"
